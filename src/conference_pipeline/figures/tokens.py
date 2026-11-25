"""Brand tokens for compiled figures.

One token set, applied by the compiler. The model never chooses a colour, a
face or a stroke width. The palette is the renderer's approved figure palette
(``generate_programmatic_figures``' allow-list) so a compiled figure and a
model-drawn one sit on the same slide without clashing.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canvas ---------------------------------------------------------------------
CANVAS_WIDTH = 1280.0
CANVAS_HEIGHT = 720.0
MARGIN = 44.0

# The rendered slide downscales this canvas to roughly a 930 px panel, so the
# legibility floor is expressed at 1280 px and is deliberately high.
FONT_FLOOR = 20.0
FONT_ABSOLUTE_FLOOR = 18.0

# Palette (renderer allow-list) ----------------------------------------------
GROUND = "#1a1a2e"
PANEL = "#16213e"
PANEL_RAISED = "#2a2a4a"
HAIRLINE = "#333344"

TEXT = "#ffffff"
TEXT_BODY = "#e0e0e0"
TEXT_MUTED = "#cccccc"

ACCENT = "#a8d8ea"      # ice blue: the default structural accent
EMPHASIS = "#ffd43b"    # amber: the one emphasised element per figure
POSITIVE = "#51cf66"
NEGATIVE = "#ff6b6b"
ALT_VIOLET = "#9775fa"
ALT_TEAL = "#4ecdc4"

# Ordered series colours. Amber is reserved for emphasis and never enters the
# rotation, so "the highlighted one" always reads as the highlighted one.
SERIES_COLORS = (ACCENT, ALT_TEAL, ALT_VIOLET, POSITIVE, NEGATIVE)

ALLOWED_COLORS = frozenset(
    {
        GROUND, PANEL, PANEL_RAISED, HAIRLINE,
        TEXT, TEXT_BODY, TEXT_MUTED,
        ACCENT, EMPHASIS, POSITIVE, NEGATIVE, ALT_VIOLET, ALT_TEAL,
        "none",
    }
)

# Type scale -----------------------------------------------------------------
FONT_TITLE = 34.0
FONT_CLAIM = 24.0
FONT_NODE = 26.0
FONT_NODE_MIN = 20.0
FONT_SUBLABEL = 20.0
FONT_EDGE = 20.0
FONT_AXIS = 21.0
FONT_TAKEAWAY = 24.0
FONT_BADGE = 19.0
FONT_STAT = 64.0

FONT_FAMILY = "Helvetica, Arial, sans-serif"

# Strokes and spacing --------------------------------------------------------
STROKE_NODE = 2.5
STROKE_EDGE = 2.5
STROKE_EMPHASIS = 4.0
STROKE_HAIRLINE = 1.5
RADIUS = 10.0

NODE_PAD_X = 18.0
NODE_PAD_Y = 14.0
NODE_SEP = 34.0
RANK_SEP = 92.0
LINE_GAP = 1.22  # multiple of font size


@dataclass(frozen=True)
class NodeStyle:
    """How one ``kind`` of topology node is drawn."""

    shape: str            # rect | round | stadium | diamond | cylinder | hexagon
    fill: str
    stroke: str
    text: str
    dashed: bool = False
    badge: str = ""       # a short glyph word drawn in the node's corner


NODE_STYLES: dict[str, NodeStyle] = {
    "input": NodeStyle("round", PANEL, ACCENT, TEXT),
    "process": NodeStyle("rect", PANEL_RAISED, ACCENT, TEXT),
    "model": NodeStyle("rect", PANEL_RAISED, ALT_TEAL, TEXT),
    "data": NodeStyle("cylinder", PANEL, ALT_VIOLET, TEXT),
    "decision": NodeStyle("diamond", PANEL, EMPHASIS, TEXT),
    "output": NodeStyle("round", PANEL, POSITIVE, TEXT),
    "frozen": NodeStyle("rect", PANEL, TEXT_MUTED, TEXT_BODY, dashed=True, badge="frozen"),
    "adapted": NodeStyle("rect", PANEL_RAISED, EMPHASIS, TEXT, badge="trained"),
    "metric": NodeStyle("stadium", PANEL, ALT_TEAL, TEXT),
    "merge": NodeStyle("round", PANEL_RAISED, EMPHASIS, TEXT),
}
DEFAULT_NODE_KIND = "process"

EDGE_STYLES: dict[str, dict[str, object]] = {
    "flow": {"stroke": ACCENT, "dash": "", "head": True},
    "feedback": {"stroke": ALT_VIOLET, "dash": "8 6", "head": True},
    "contrast": {"stroke": NEGATIVE, "dash": "2 7", "head": False},
    "dashed": {"stroke": TEXT_MUTED, "dash": "8 6", "head": True},
    "parallel": {"stroke": ALT_TEAL, "dash": "4 6", "head": False},
}
DEFAULT_EDGE_KIND = "flow"

# Grounding badges -----------------------------------------------------------
SCHEMATIC_BADGE_TEXT = "schematic - illustrates the claim, not measured data"
HYBRID_BADGE_TEXT = "schematic layout, measured callouts"
