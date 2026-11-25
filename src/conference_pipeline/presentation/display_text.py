"""Text that reaches a slide has to read as text.

Corpus titles and ledger notes arrive as their authors typed them: LaTeX in a
paper title, Markdown in a contribution note, an event-kind prefix on a talk
title. Three defects shipped on the ICML 2026 canary decks because each cleaner
handled one of these and not its neighbours:

* ``$\\texttt{FlashSchNet}$`` rendered as ``textttFlashSchNet``. The converter
  kept an unknown macro's *name* (``\\texttt``) and dropped its braces, so a text
  macro leaked its own name into the slide.
* ``[OpenReview](https://...)`` appeared verbatim on a resources slide, because
  the bookend cleaner stripped emphasis markers but not links.
* A source label was cut at its length limit on a connector ("... Inference &").

This module is the one place those rules live. It has no dependencies on the
rest of the pipeline so the ledger builder, the slide renderers and the lint
can all import it without a cycle.

Two kinds of function: *conversion* (``latex_to_text``, ``strip_markdown``) fixes
text as it is made, and *repair* (``repair_leaked_macro_names``) fixes text that
was already made wrong and is sitting in a built ledger -- re-running the chain
to regenerate it would cost far more than repairing it at display time.
"""

from __future__ import annotations

import html
import re

# Macros that wrap text. Their argument is the content; the macro itself means
# nothing on a slide.
TEXT_MACROS = (
    "texttt", "textbf", "textit", "textsc", "textrm", "textsf", "textnormal", "emph",
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "text", "mbox",
    "operatorname", "boldsymbol",
)
_TEXT_MACRO_RE = re.compile(r"\\(?:" + "|".join(TEXT_MACROS) + r")\s*\{([^{}]*)\}")
_ESCAPED_SPECIAL_RE = re.compile(r"\\([&%#$_{}])")

# What is left of a text macro once its backslash and braces are gone and the
# next token is glued to it: ``textttFlashSchNet``. Only a macro name followed
# directly by an uppercase letter or digit counts, so ordinary words are safe.
_LEAKED_MACRO_RE = re.compile(
    r"(?<![A-Za-z\\])(?:" + "|".join(sorted(TEXT_MACROS, key=len, reverse=True)) + r")(?=[A-Z0-9])"
)

_LATEX_SYMBOLS = {
    "oslash": "⊘", "nrightarrow": "↛", "rightarrow": "→", "to": "→",
    "leftarrow": "←", "leftrightarrow": "↔", "times": "×", "cdot": "·",
    "infty": "∞", "approx": "≈", "leq": "≤", "geq": "≥", "pm": "±",
    "emptyset": "∅", "varnothing": "∅", "alpha": "α", "beta": "β",
    "gamma": "γ", "delta": "δ", "epsilon": "ε", "lambda": "λ",
    "mu": "μ", "pi": "π", "sigma": "σ", "theta": "θ", "phi": "φ",
    "Delta": "Δ", "Sigma": "Σ", "star": "⋆", "dagger": "†",
}
_SUPERSCRIPTS = str.maketrans("0123456789+-n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ")
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

_INLINE_MATH_RE = re.compile(r"\$(?=[^\s\d$])([^$\n]{1,60}?)(?<=\S)\$")

_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:[^()\s]|\([^()]*\))*\)")
# Emphasis is removed only as a PAIR around text. A lone star is content ("A* search",
# "p < 0.05*"), and stripping every star would turn it into a different word.
_MARKDOWN_PAIRS = (
    re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"),
    re.compile(r"__(?=\S)(.+?)(?<=\S)__"),
    re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]*?)(?<=[^\s*])\*(?![\w*])"),
    re.compile(r"`([^`\n]+)`"),
    re.compile(r"~~(?=\S)(.+?)(?<=\S)~~"),
)
# "*Title* — , [OpenReview](...)" leaves a separator with nothing on one side.
_ORPHAN_SEPARATOR_RE = re.compile(r"\s*[—–-]\s*(?=[,;])")

# Words and marks a label must not end on; the figure-label cutter's own list
# (``structured_video.SVG_FRAGMENT_END_WORDS``) is the authority for words, this
# adds the typographic connectors.
DANGLING_MARKS = frozenset({"&", "+", "/", "|", "-", "–", "—", ":", ",", ";", "("})

_EVENT_KIND_PREFIX_RE = re.compile(
    r"^\s*(?:poster|oral|spotlight|workshop|tutorial|keynote|invited talk|talk|demo)\s*:\s+", re.I
)


def unwrap_text_macros(text: str) -> str:
    """``\\texttt{X}`` -> ``X``, repeatedly so nested wrappers unwrap too."""

    previous = None
    while previous != text:
        previous = text
        text = _TEXT_MACRO_RE.sub(r"\1", text)
    return text


def latex_to_text(math: str) -> str:
    """The plain text a piece of inline math stands for."""

    text = unwrap_text_macros(str(math or ""))
    text = _ESCAPED_SPECIAL_RE.sub(r"\1", text)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: _LATEX_SYMBOLS.get(m.group(1), m.group(1)), text)
    text = re.sub(
        r"\^\{([0-9+\-n]+)\}|\^([0-9+\-n])",
        lambda m: (m.group(1) or m.group(2)).translate(_SUPERSCRIPTS),
        text,
    )
    text = re.sub(r"_\{([0-9]+)\}|_([0-9])", lambda m: (m.group(1) or m.group(2)).translate(_SUBSCRIPTS), text)
    return text.replace("{", "").replace("}", "").strip()


def repair_leaked_macro_names(text: str) -> str:
    """Remove a text macro's name that an earlier cleaner left glued to a word.

    ``textttFlashSchNet`` -> ``FlashSchNet``. For text that is already built; new
    text should never need it because ``latex_to_text`` unwraps the macro.
    """

    return _LEAKED_MACRO_RE.sub("", str(text or ""))


def clean_display_text(text: str) -> str:
    """Strip typesetting (inline math, text macros, PDF line-break hyphens) from a title or note."""

    if not text:
        return text
    cleaned = unwrap_text_macros(text)
    cleaned = _INLINE_MATH_RE.sub(lambda m: latex_to_text(m.group(1)), cleaned)
    cleaned = repair_leaked_macro_names(cleaned)
    # "sam- ples" -> "samples", but keep suspended hyphens ("short- and long-term").
    cleaned = re.sub(r"(?<=[a-z])- (?!(?:and|or|nor|to|vs|versus)\b)(?=[a-z])", "", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def strip_markdown(text: str) -> str:
    """Markdown as the plain text a slide should show.

    A link keeps its text and loses its URL; an image keeps its alt text; emphasis
    and code markers go; a separator orphaned by a removed field goes too.
    """

    cleaned = html.unescape(str(text or ""))
    cleaned = _MARKDOWN_IMAGE_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    for pair in _MARKDOWN_PAIRS:
        cleaned = pair.sub(r"\1", cleaned)
    cleaned = _ORPHAN_SEPARATOR_RE.sub("", cleaned)
    cleaned = re.sub(r"([,;])(?:\s*[,;])+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([,;.])", r"\1", cleaned)
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned)
    return " ".join(cleaned.split())


def strip_event_kind_prefix(text: str) -> str:
    """``Poster: Walrus: ...`` -> ``Walrus: ...`` (the chip carries its own kind tag)."""

    return _EVENT_KIND_PREFIX_RE.sub("", str(text or ""))


def is_dangling_word(word: str, fragment_words: frozenset[str] = frozenset()) -> bool:
    """True for a connector mark or function word a label must not end on.

    A single uppercase letter is a label ("Variant A"), never a stopword.
    """

    stripped = word.strip(".,;:!?")
    if not stripped:
        return True
    if word in DANGLING_MARKS or stripped in DANGLING_MARKS:
        return True
    if len(stripped) == 1 and stripped.isalpha() and stripped.isupper():
        return False
    return stripped.lower() in fragment_words


def ends_dangling(text: str, fragment_words: frozenset[str] = frozenset()) -> bool:
    """True when a multi-word label ends on a connector mark or function word."""

    words = str(text or "").split()
    return len(words) > 1 and is_dangling_word(words[-1], fragment_words)


def trim_dangling_end(text: str, *, fragment_words: frozenset[str] = frozenset()) -> str:
    """Drop trailing connectors and function words so a label ends on content.

    ``4th Structured Probabilistic Inference &`` -> ``4th Structured Probabilistic
    Inference``. ``fragment_words`` is the caller's stopword list (lower-case).
    """

    words = str(text or "").split()
    while len(words) > 1 and is_dangling_word(words[-1], fragment_words):
        words.pop()
    return " ".join(words).rstrip(".,;:-\u2013&").strip()
