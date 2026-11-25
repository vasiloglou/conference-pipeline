#!/usr/bin/env python3
"""Idempotently inject deck-scoped AI4Physics Reveal.js polish CSS.

This is the third step of the AI4Physics render pipeline (Phase 40-03):

    1. adapt_ai4physics_slides_to_revealjs.py  -- author markdown -> canonical intermediate
    2. md_to_revealjs.py                       -- UNMODIFIED shared renderer -> presentation.html
    3. apply_ai4physics_revealjs_polish.py      -- THIS SCRIPT -- deck-scoped visual polish

The shared renderer (scripts/presentations/md_to_revealjs.py) is intentionally left
untouched by this script and by this pipeline. Two cosmetic issues in its generic
output only affect this specific deck's content lengths:

  1. Every brand mark (title logo, decorative art, content "bug") is emitted TWICE
     by the renderer -- once as a CSS pseudo-element background (::before/::after)
     and once as a real <img>. Both are visible, producing a doubled/offset mark.
     The backgrounds are correctly positioned; this script hides the redundant imgs.

  2. The title slide's long subtitle and the one section-kind divider's long body
     paragraph exceed the fixed 720px slide height (`overflow: hidden` on
     `.reveal .slides section`), clipping content off the bottom for THIS deck's
     particular title/body lengths.

Rather than hand-editing presentation.html (which a re-render would silently
discard), this script inserts a single `<style id="ai4physics-polish">` block
right before `</head>`. It is idempotent: re-running it replaces any prior
polish block instead of stacking duplicates, so it is safe to run after every
re-render.

This script ALSO strips internal figure-provenance captions from the inlined
SVGs. Several Phase 38.5 figure SVGs bake a bottom-of-figure `<text>` caption
that starts with "Create-our-own" and records internal sourcing scaffolding
(e.g. "Source: iclr2026/decks/cost_actual_summary.json ...", "metrics-pending",
"figures for the reconciliation pass", "no fabricated numbers"). These are
production notes, not audience content, so they are removed from the RENDER
here. The source SVG assets under `cross-conference-talks/ai4physics2026/assets/` are left
untouched (editing figures is Phase 38.5's domain and the Google Slides sync is
HELD), so the strip is re-applied on every render as the third pipeline step.

Usage:
    .venv/bin/python scripts/presentations/apply_ai4physics_revealjs_polish.py \\
        cross-conference-talks/ai4physics2026/decks/ai4physics-talk/presentation.html
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

POLISH_STYLE_ID = "ai4physics-polish"
SLIDE_NUMBER_SCRIPT_ID = "ai4physics-slide-number"

# Internal figure-provenance / production-scaffolding captions baked into the
# Phase 38.5 SVGs, rendered as muted footnote <text> elements at the bottom of a
# figure. They are single flat <text> elements (plain text / character entities
# only, no nested tags), so `[^<]*` for the inner content is exact and safe;
# `[^<]` also spans newlines, covering any wrapped caption.
_SVG_TEXT_RE = re.compile(r"<text\b[^>]*>([^<]*)</text>")

# A caption is INTERNAL scaffolding (strip it) when its text references a
# production artifact or the figure's own construction. These signals never
# appear in audience-facing figure labels (verified against every muted-gray
# <text> in the real deck): repo markdown files (narrative.md, THESIS.md,
# 09_genai_in_science.md), a `conferences/...` path, a leading "Source(s):",
# the "Create-our-own" provenance prefix, "no [paper] figure cropped" /
# "original redraw" figure-construction notes, "no fabricated numbers", or the
# "not derived from the conference analysis" disclaimer. Honest audience
# annotations that only describe the figure's content (e.g. "Qwen 122B option
# wants a ~$5,000 Mac (speaker estimate)", "asked from the stage") match NONE of
# these and are preserved.
_PROVENANCE_SIGNALS = (
    re.compile(r"Create-our-own", re.IGNORECASE),
    re.compile(r"[\w-]+\.md\b"),
    re.compile(r"conferences/"),
    re.compile(r"^\s*sources?:", re.IGNORECASE),
    re.compile(r"no (?:paper )?figure cropped", re.IGNORECASE),
    re.compile(r"original redraw", re.IGNORECASE),
    re.compile(r"no fabricated numbers", re.IGNORECASE),
    re.compile(r"not derived from the conference analysis", re.IGNORECASE),
)


def _is_provenance_caption(text_content: str) -> bool:
    return any(sig.search(text_content) for sig in _PROVENANCE_SIGNALS)

_POLISH_CSS = """<style id="ai4physics-polish">
/* AI4Physics Reveal.js delivery polish (Phase 40-03) -- deck-scoped overrides
   layered on top of the UNMODIFIED shared renderer output. Regenerate via:
     .venv/bin/python scripts/presentations/adapt_ai4physics_slides_to_revealjs.py \\
       --input cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md \\
       --canonical-out cross-conference-talks/ai4physics2026/decks/ai4physics-talk/_canonical_slides.md
     .venv/bin/python scripts/presentations/md_to_revealjs.py \\
       --input cross-conference-talks/ai4physics2026/decks/ai4physics-talk/_canonical_slides.md \\
       --output cross-conference-talks/ai4physics2026/decks/ai4physics-talk/presentation.html \\
       --slide-mapping cross-conference-talks/ai4physics2026/decks/ai4physics-talk/presentation_slide_mapping.json --verbose
     .venv/bin/python scripts/presentations/apply_ai4physics_revealjs_polish.py \\
       cross-conference-talks/ai4physics2026/decks/ai4physics-talk/presentation.html
   See 40-03-SUMMARY.md for screenshots confirming each fix below. */

/* 1. Remove doubled brand marks. The renderer emits BOTH a CSS pseudo-element
   background (::before/::after, correctly positioned) AND a real <img> (object-fit
   centered) for the title logo, decorative art, and content "bug" -- producing a
   visibly doubled/offset mark on every slide. Keep the background, hide the img. */
.reveal .slides section .title-logo,
.reveal .slides section .decorative-art,
.reveal .slides section .content-brand-bug {
  display: none !important;
}

/* 2. Title slide: this deck's H1 + long subtitle (title-xlong variant) clipped off
   the bottom of the fixed 720px slide. Tighten top margin and shrink both lines so
   the full subtitle ("...how we learned to read them at scale") is visible with a
   little bottom margin, across all title-length variants. */
.reveal .slides section.slide-title .title-content {
  margin-top: 60px !important;
  max-width: 66% !important;
}
.reveal .slides section.slide-title h1 {
  font-size: 1.4em !important;
  line-height: 1.15 !important;
}
.reveal .slides section.slide-title h2 {
  font-size: 1.0em !important;
  line-height: 1.15 !important;
  margin-top: 0.3em !important;
}

/* 3. Section divider: the one section-kind divider ("What did we learn from the
   latest NeurIPS, ICLR, ICML?") has an H1 + subtitle + body paragraph ("How to
   read this section: ...") that clipped off the bottom. Shrink the H1 slightly,
   tighten the top margin, and size the body paragraph -- the four short
   native-divider slides (H1[+H2] only, no body paragraph) stay well within bounds
   at this size and are unaffected in practice. */
.reveal .slides section.slide-section h1 {
  font-size: 1.35em !important;
}
.reveal .slides section.slide-section .section-inner {
  margin-top: 1% !important;
}
.reveal .slides section.slide-section p {
  font-size: 0.5em !important;
  line-height: 1.3 !important;
  margin-top: 0.35em !important;
}

/* 4. Content-slide subtitle. Each findings/content slide's `## ...` line renders
   as an <h2> under the H3 title, but the shared renderer never styles content
   <h2>, so it inherits Reveal's oversized default (larger than the title above
   it). Long subtitles -- e.g. slide 42 "The useful scientific-agent loop is
   reproduce and execute first, then critique; language-level plausibility is
   not enough" and slide 47 -- then push the bullets off the bottom of the
   fixed-height slide. Shrink to a true subtitle size (below the H3 title's
   1.15em, above the 0.58em bullets) with tight leading so nothing clips. */
.reveal .slides section.slide-content h2,
.reveal .slides section.slide-content-with-figure h2 {
  font-size: 0.72em !important;
  line-height: 1.2 !important;
  margin: 0.1em 0 0.4em 0 !important;
}

/* 5. Prose-heavy content slides. A few content slides carry running paragraphs
   plus a blockquote instead of bullets (e.g. slide 8 "Are conferences publishing
   in the right language?": five paragraphs + a quote). At the shared renderer's
   0.60em paragraph size these run past the bottom of the fixed 720px slide.
   Tighten paragraph and blockquote text on content slides so the full argument
   fits; bullet-only slides have no <p> and are unaffected. */
.reveal .slides section.slide-content p {
  font-size: 0.5em !important;
  line-height: 1.28 !important;
  margin: 0.12em 0 !important;
}
.reveal .slides section.slide-content blockquote {
  font-size: 0.46em !important;
  padding: 0.15em 0.5em !important;
  margin: 0.25em 0 !important;
}
</style>"""

# Root cause (Phase quick-260714-dsu): presentation.html has 72 top-level
# <section> elements -- 49 authored slides + 23 auto-generated figure-zoom
# companions (see md_to_revealjs.py, unmodified/shared). Reveal's default
# on-screen counter shows the flat 1..72 section index, so e.g. authored
# slide 8 (two sections: main + zoom companion) displays as "11" / "12"
# instead of "8 / 49" -- a mismatch against presenter_cheatsheet.md,
# narrative.md, DELIVERY.md, and Google Slides, all of which use the
# authored 1..49 position. The shared renderer already stamps every
# section with data-markdown-slide="{authored_position}"; this script
# reconfigures Reveal's slideNumber to read that attribute instead of the
# flat index.
#
# This block is injected at the end of <body>, after Reveal.initialize (the
# last script before </body>), so the `Reveal` global already exists. It
# honors the existing `?hidecontrols` behavior (early-returns without
# touching Reveal.configure when the token is present) and derives the
# total slide count from the DOM (max data-markdown-slide) rather than a
# hardcoded literal, so it stays deck-agnostic.
_SLIDE_NUMBER_SCRIPT = """<script id="ai4physics-slide-number">
(function () {
  // Mirrors the hidecontrols check in the Reveal.initialize script above:
  // when present, the counter is meant to stay hidden -- do not re-enable
  // slideNumber under hidecontrols.
  var hideControls = window.location.search.indexOf('hidecontrols') !== -1;
  if (hideControls) {
    return;
  }

  function configureAuthoredSlideNumber() {
    var sections = document.querySelectorAll(
      '.reveal .slides section[data-markdown-slide]'
    );
    var total = 0;
    for (var i = 0; i < sections.length; i++) {
      var value = parseInt(sections[i].getAttribute('data-markdown-slide'), 10);
      if (!isNaN(value) && value > total) {
        total = value;
      }
    }

    Reveal.configure({
      slideNumber: function (slide) {
        var position = parseInt(slide.getAttribute('data-markdown-slide'), 10);
        if (isNaN(position)) {
          position = Reveal.getSlidePastCount(slide) + 1;
        }
        return [position, '/', total];
      },
    });
  }

  if (typeof Reveal !== 'undefined' && Reveal.isReady && Reveal.isReady()) {
    configureAuthoredSlideNumber();
  } else if (typeof Reveal !== 'undefined') {
    Reveal.on('ready', configureAuthoredSlideNumber);
  }
})();
</script>"""


def strip_provenance_captions(html: str) -> tuple[str, int]:
    """Remove internal figure-provenance / production-scaffolding <text> captions.

    Removes only SVG `<text>` elements whose content trips a `_PROVENANCE_SIGNALS`
    marker (repo artifact reference or figure-construction note); audience-facing
    figure labels and honest content annotations are preserved. Returns the
    cleaned HTML and the number of caption elements removed. Idempotent: once
    removed there is nothing left to match on a re-run.
    """
    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        if _is_provenance_caption(match.group(1)):
            count += 1
            return ""
        return match.group(0)

    cleaned = _SVG_TEXT_RE.sub(_replace, html)
    return cleaned, count


def apply_polish(html: str) -> str:
    """Idempotently insert/replace the polish <style> block right before </head>."""
    existing_block = re.compile(
        r'<style id="' + re.escape(POLISH_STYLE_ID) + r'">.*?</style>\s*',
        re.DOTALL,
    )
    html = existing_block.sub("", html)
    if "</head>" not in html:
        raise ValueError("presentation.html has no </head> tag to inject polish before")
    return html.replace("</head>", _POLISH_CSS + "\n</head>", 1)


def inject_slide_number(html: str) -> str:
    """Idempotently insert/replace the slide-number <script> right before </body>.

    Mirrors `apply_polish()`'s remove-then-reinsert idempotency contract: any
    prior `<script id="ai4physics-slide-number">` block is stripped first, then
    a fresh block is inserted immediately before the single `</body>` tag.
    """
    existing_block = re.compile(
        r'<script id="' + re.escape(SLIDE_NUMBER_SCRIPT_ID) + r'">.*?</script>\s*',
        re.DOTALL,
    )
    html = existing_block.sub("", html)
    if "</body>" not in html:
        raise ValueError("presentation.html has no </body> tag to inject slide-number script before")
    return html.replace("</body>", _SLIDE_NUMBER_SCRIPT + "\n</body>", 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "html_path",
        type=Path,
        help="Path to the rendered presentation.html to polish in place",
    )
    args = parser.parse_args(argv)

    if not args.html_path.exists():
        print(f"error: {args.html_path} does not exist", file=sys.stderr)
        return 1

    html = args.html_path.read_text("utf-8")
    html, captions_removed = strip_provenance_captions(html)
    html = apply_polish(html)
    polished = inject_slide_number(html)
    args.html_path.write_text(polished, "utf-8")
    print(
        f"Applied '{POLISH_STYLE_ID}' polish block and '{SLIDE_NUMBER_SCRIPT_ID}' "
        f"slide-number script to {args.html_path} "
        f"(stripped {captions_removed} internal provenance caption(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
