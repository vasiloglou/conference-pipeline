"""Measured text: wrapping, fitting and block sizing.

Every width here comes from ``svg_geometry.estimate_text_width`` -- the same
metric the geometry gate measures the emitted SVG with, so layout and checking
cannot disagree. It is a glyph-width table plus a render safety factor, which
makes it deterministic on any machine and independent of installed fonts.
"""

from __future__ import annotations

from dataclasses import dataclass

from conference_pipeline.figures import tokens
from conference_pipeline.presentation.svg_geometry import estimate_text_width


def measure(text: str, size: float, *, bold: bool = False) -> float:
    """Rendered advance width of ``text`` at ``size``."""

    return estimate_text_width(text, size, bold=bold)


def line_height(size: float) -> float:
    return size * tokens.LINE_GAP


@dataclass(frozen=True)
class TextBlock:
    """Wrapped text with its measured box.

    ``source`` is the caller's original string, before wrapping and before any
    hyphenation. It exists so the manifest can answer "did this label reach the
    canvas?" without having to un-wrap ``lines`` -- a question that matters
    because a layout may legitimately shrink or re-anchor a label, and used to
    be able to drop one entirely with no trace anywhere.
    """

    lines: tuple[str, ...]
    size: float
    bold: bool
    width: float
    height: float
    source: str = ""
    # Words this block had to split mid-word, with the fragments it produced.
    breaks: tuple[tuple[str, ...], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.lines


EMPTY_BLOCK = TextBlock(lines=(), size=0.0, bold=False, width=0.0, height=0.0, source="")


MIN_WORD_FRAGMENT = 4


def _break_long_word(word: str, size: float, max_width: float, *, bold: bool) -> list[str]:
    """Split a single word that cannot fit, at the widest prefix that does.

    Only reached for tokens with no spaces (a URL, a hyphen-free identifier);
    ordinary prose never gets here because ``wrap`` keeps words whole.

    No fragment shorter than ``MIN_WORD_FRAGMENT``. "Counterfactual" was being
    rendered as "Count- / erfactu- / al" -- three fragments, one of them two
    letters, in a shipped figure that passed every gate because nothing looked
    at hyphenation. A two-letter fragment is not a syllable; it is a defect.
    """

    pieces: list[str] = []
    current = ""
    for char in word:
        candidate = current + char
        if len(current) >= MIN_WORD_FRAGMENT and measure(candidate + "-", size, bold=bold) > max_width:
            pieces.append(current + "-")
            current = char
        else:
            current = candidate
    if current:
        if pieces and len(current) < MIN_WORD_FRAGMENT:
            # Give the runt back to the piece before it rather than leaving a
            # two-letter line of its own.
            pieces[-1] = pieces[-1][:-1] + current
        else:
            pieces.append(current)
    return pieces or [word]


MIN_COLUMN = 40.0


def wrap(text: str, size: float, max_width: float, *, bold: bool = False) -> list[str]:
    """Greedy word wrap that never splits a word that fits on a line of its own."""

    return wrap_with_breaks(text, size, max_width, bold=bold)[0]


def wrap_with_breaks(
    text: str, size: float, max_width: float, *, bold: bool = False
) -> tuple[list[str], list[tuple[str, ...]]]:
    """Wrapped lines, plus the words this call had to split MID-WORD.

    Splitting at a hyphen the author wrote ("AVA-Bench") is ordinary wrapping.
    Inserting a hyphen inside a word is not, and the two are indistinguishable
    once you only have the output lines -- which is why the breaks are recorded
    where they happen rather than detected afterwards.
    """

    # A caller that computes a column from a distance can hand us zero or a
    # negative number; wrapping into it would break every word into single
    # characters instead of failing, which is how a stack of one-letter lines
    # reached the canvas edge.
    max_width = max(MIN_COLUMN, float(max_width))
    inserted: list[tuple[str, ...]] = []
    words: list[str] = []
    for token in str(text).split():
        # "Concept-separated" wraps after the hyphen, never inside a syllable.
        if "/" in token[1:-1] and measure(token, size, bold=bold) > max_width:
            # A slash in an authored term is a natural line break. Keeping
            # "Instrument/anatomy" as one unbreakable word made an otherwise
            # feasible architecture column wider than its whole canvas share.
            pieces = token.split("/")
            words.extend(f"{piece}/" for piece in pieces[:-1])
            words.append(pieces[-1])
        elif "-" in token[1:-1] and measure(token, size, bold=bold) > max_width:
            pieces = token.split("-")
            words.extend(f"{piece}-" for piece in pieces[:-1])
            words.append(pieces[-1])
        else:
            words.append(token)
    if not words:
        return [], inserted
    lines: list[str] = []
    current = ""
    for word in words:
        joiner = "" if current.endswith(("-", "/")) else " "
        candidate = f"{current}{joiner}{word}".strip()
        if current and measure(candidate, size, bold=bold) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
        # A single word wider than the column is NOT hyphenated: the caller
        # sizes its box from the measured block, so the box grows instead. Only
        # a runaway token (a URL, an identifier) past 1.6x the column is split,
        # because no box could hold it.
        if " " not in current and current.count("-") == 0 and measure(current, size, bold=bold) > max_width * 1.6:
            pieces = _break_long_word(current, size, max_width, bold=bold)
            if len(pieces) > 1:
                inserted.append(tuple(pieces))
            lines.extend(pieces[:-1])
            current = pieces[-1]
    if current:
        lines.append(current)
    return lines, inserted


def fit(
    text: str,
    *,
    max_width: float,
    max_lines: int = 3,
    size: float = tokens.FONT_NODE,
    min_size: float = tokens.FONT_NODE_MIN,
    bold: bool = False,
    step: float = 1.0,
) -> TextBlock:
    """Wrap ``text`` into at most ``max_lines``, shrinking towards ``min_size``.

    Never truncates: if the text still needs more lines at ``min_size``, the
    block keeps every line and reports its true height, and the caller (the
    layout) grows the node instead. Truncated text is a defect; a taller node
    is a layout decision.
    """

    if not str(text).strip():
        return EMPTY_BLOCK
    current = float(size)
    lines, inserted = wrap_with_breaks(text, current, max_width, bold=bold)
    # Shrink for a word that will not fit the column BEFORE any thought of
    # breaking it: the node's own width floor usually leaves more room than the
    # budget-derived column the wrapper was handed, and shrinking two points is
    # invisible where a mid-word hyphen is not.
    def widest_word_at(size: float) -> float:
        return max(
            (measure(part, size, bold=bold)
             for word in str(text).split()
             for part in ([f"{piece}/" for piece in word.split("/")[:-1]]
                          + [word.split("/")[-1]] if "/" in word[1:-1] else [word])),
            default=0.0,
        )

    widest_word = widest_word_at(current)
    while widest_word > max_width and current - step >= min_size:
        current -= step
        widest_word = widest_word_at(current)
        lines, inserted = wrap_with_breaks(text, current, max_width, bold=bold)
    while len(lines) > max_lines and current - step >= min_size:
        current -= step
        lines, inserted = wrap_with_breaks(text, current, max_width, bold=bold)
    width = max((measure(line, current, bold=bold) for line in lines), default=0.0)
    return TextBlock(
        lines=tuple(lines),
        size=current,
        bold=bold,
        width=width,
        height=len(lines) * line_height(current),
        source=str(text),
        breaks=tuple(inserted),
    )


def fit_to_box(
    text: str,
    *,
    box_width: float,
    box_height: float,
    size: float,
    min_size: float,
    bold: bool = False,
    step: float = 1.0,
) -> TextBlock:
    """Largest size at or below ``size`` whose wrapped block fits the box."""

    if not str(text).strip():
        return EMPTY_BLOCK
    current = float(size)
    while True:
        lines, inserted = wrap_with_breaks(text, current, box_width, bold=bold)
        height = len(lines) * line_height(current)
        widest = max((measure(line, current, bold=bold) for line in lines), default=0.0)
        if (height <= box_height and widest <= box_width) or current - step < min_size:
            return TextBlock(
                lines=tuple(lines), size=current, bold=bold, width=widest, height=height,
                source=str(text), breaks=tuple(inserted),
            )
        current -= step


def block_for_node(
    label: str,
    sublabel: str = "",
    *,
    max_width: float,
    size: float = tokens.FONT_NODE,
    min_size: float = tokens.FONT_NODE_MIN,
    sub_size: float = tokens.FONT_SUBLABEL,
    max_lines: int = 3,
) -> tuple[TextBlock, TextBlock]:
    """Title and optional sublabel blocks for one node, measured together."""

    title = fit(label, max_width=max_width, max_lines=max_lines, size=size, min_size=min_size, bold=True)
    sub = (
        fit(sublabel, max_width=max_width, max_lines=2, size=sub_size, min_size=tokens.FONT_FLOOR)
        if str(sublabel).strip()
        else EMPTY_BLOCK
    )
    return title, sub
