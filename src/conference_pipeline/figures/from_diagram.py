"""Turn a digest diagram sketch into a figure-compiler spec.

The executive-summary decks are authored by the digest chain, which describes
each figure as ``{kind, caption, nodes}`` -- a diagram KIND plus three to five
short labels -- and used to hand-draw it with a renderer of its own. That meant
two live drawing paths in one conference: the topic decks went through the
figure compiler and its geometry gate, while the executive decks' figures were
never gated at all.

This module is the bridge. It maps a sketch onto a compiler archetype that
needs nothing more than flat labels, so every deck in a workspace renders
through one compiler and is measured by one gate.

What it deliberately does NOT do: invent numbers. A sketch carries labels, so
the specs it produces are schematic, and the compiler's own rule that a
quantitative archetype must cite ledger rows keeps it that way.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

# Every archetype here takes flat labels. A sketch has no rows, columns, axes
# or values, so mapping it onto an archetype that needs them would mean
# inventing the missing half.
DIAGRAM_KIND_TO_ARCHETYPE: dict[str, str] = {
    "pipeline": "flow",
    "mechanism": "flow",
    "flow": "flow",
    "architecture": "layered_stack",
    "stack": "layered_stack",
    "comparison": "contrast_pair",
    "ladder": "ranked_list",
    "checklist": "ranked_list",
    "list": "ranked_list",
    "matrix": "hub_spoke",
    "grid": "hub_spoke",
    # Added 2026-09-15. The vision judge failed 11 executive figures for one
    # reason: the sketch vocabulary had no word for a branch, a loop, or a
    # two-axis map, so authors wrote "mechanism" and a genuine fork rendered as
    # one straight pipeline -- "Cheap early exit" drawn as a mandatory stage
    # before "Escalation". These three carry a label CONVENTION (documented on
    # each builder) so the sketch still states everything the figure shows.
    "branch": "branching_paths",
    "routing": "branching_paths",
    "fork": "branching_paths",
    "loop": "loop",
    "cycle": "loop",
    "quadrant": "quadrant_map",
}
DEFAULT_ARCHETYPE = "ranked_list"

# "Old: parameter count" / "New: peak device memory" -- a sketch states a
# comparison by prefixing its labels, which is what makes contrast_pair usable.
# The same prefix names a branch: "Easy: exit cheaply" / "Hard: escalate".
# The separator is a colon, or a dash SET OFF BY SPACES. A bare hyphen is not
# one: "Scale-first: one embedding" names the side "Scale-first", and reading
# the hyphen as the separator produced a column titled "Scale" over the item
# "first: one embedding" -- which the vision judge passed.
# Digits are allowed: "Direct 2D VLA: ..." vs "3D-first: ..." was not read as
# two sides and rendered as a four-step numbered list (CVPR 2026 executive deck).
_SIDE_RE = re.compile(
    r"^\s*(?P<side>[A-Za-z0-9][A-Za-z0-9 /-]{0,22}?)\s*(?::|\s[–-]\s)\s*(?P<rest>.+)$"
)
# A quadrant sketch names its axes as "X: <low> | <high>" and "Y: <low> | <high>"
# and places labels by corner. Corners, not coordinates: a sketch positions
# qualitatively, and a number here would be invented precision.
_AXIS_RE = re.compile(
    r"^\s*(?P<axis>[xXyY])(?:\s*axis)?\s*[:–-]\s*(?P<low>[^|]+?)\s*\|\s*(?P<high>.+?)\s*$"
)
_CORNER_RE = re.compile(
    r"^\s*(?P<corner>top-left|top-right|bottom-left|bottom-right)\s*[:–-]\s*(?P<label>.+?)\s*$",
    re.IGNORECASE,
)
_CORNER_XY: dict[str, tuple[float, float]] = {
    "top-left": (0.25, 0.75),
    "top-right": (0.75, 0.75),
    "bottom-left": (0.25, 0.25),
    "bottom-right": (0.75, 0.25),
}


# A takeaway is a headline, not a sentence, and the compiler enforces that.
# Words that only ever sit mid-phrase. A headline ending on one of these reads
# as truncated, which is exactly what the renderer's own gate refuses.
FRAGMENT_END_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "because", "between", "but", "by",
    "for", "from", "in", "into", "is", "nor", "of", "on", "onto", "or", "per",
    "the", "through", "to", "toward", "towards", "under", "upon", "using",
    "via", "with", "within",
})
# Words that can legitimately end a finished sentence, but only ever signal a
# cut when a TRIM produced them: "... the cache instead" lost its "of X".
# Applied only after truncation, so a headline that was already short keeps
# its own wording.
TRIM_TRAILING_WORDS = frozenset({
    "instead", "rather", "compared", "relative", "versus", "vs", "than",
    "before", "after", "across", "beyond", "without", "against",
    # A clause opener or auxiliary left behind by the cut: "... to a rank that
    # can" and "... gates before it" (ICML 2026 executive deck, vision judge).
    "that", "which", "who", "whose", "when", "where", "while", "if", "it",
    "its", "they", "their", "can", "could", "will", "would", "must",
    "should", "may", "might", "be", "been", "being", "has", "have", "had",
    "do", "does",
})
CLAUSE_OPENERS = frozenset({
    "that", "which", "who", "whose", "before", "after", "until", "unless",
    "because", "when", "where", "while", "if", "so",
})
PHRASE_PREPOSITIONS = frozenset({
    "in", "of", "for", "on", "to", "from", "with", "by", "at", "via", "across",
    "into", "under", "over", "through", "within", "without", "between",
})
TAKEAWAY_MAX_WORDS = 12
# A figure title is a heading, not the caption sentence.
TITLE_MAX_WORDS = 10
# A hub label sits inside a small central node. A sketch's caption is a
# sentence, so it is usually the wrong thing to put there.
HUB_LABEL_MAX_CHARS = 28


def _headline(value: str, max_words: int = TAKEAWAY_MAX_WORDS) -> str:
    """Trim prose to a headline that does not read as a cut-off sentence.

    Trimming on a word boundary is not enough: "... instead of the" ends on an
    article and the geometry gate refuses it, correctly. Drop trailing words
    that only ever appear mid-phrase, and prefer a complete leading clause when
    the text has one.
    """

    text = " ".join(str(value).split())
    if not text:
        return ""
    words = text.split()
    trimmed = False
    if len(words) > max_words:
        # A leading clause that fits is a real phrase; a slice of a sentence is not.
        for separator in (":", ";", ","):
            head = text.split(separator)[0].strip()
            if head and len(head.split()) <= max_words:
                return head
        words = words[:max_words]
        trimmed = True
        # A cut inside a subordinate clause leaves half of it ("... gates
        # before it becomes"); end the headline where that clause began.
        openers = [index for index, word in enumerate(words)
                   if index >= 3 and word.lower() in CLAUSE_OPENERS]
        if openers:
            words = words[:openers[-1]]
        # Otherwise a cut inside a trailing prepositional phrase leaves its
        # object unfinished ("... physical guarantees in neural"); drop it.
        tail = [] if openers else [index for index, word in enumerate(words)
                if index >= 4 and index >= len(words) - 3 and word.lower() in PHRASE_PREPOSITIONS]
        if tail:
            words = words[:tail[-1]]
    # Dropping a trailing word is itself a truncation, so once anything has
    # been removed the stricter list applies to whatever it exposed: "... the
    # cache instead of the" loses "the" and "of", and the "instead" left
    # behind reads just as cut off as they did.
    while words:
        tail = words[-1].lower().strip(",;:-")
        stop = FRAGMENT_END_WORDS | (TRIM_TRAILING_WORDS if trimmed else frozenset())
        if tail not in stop:
            break
        words.pop()
        trimmed = True
    return " ".join(words).rstrip(",;:-")


def _short_label(*candidates: str) -> str:
    """The first candidate short enough to sit inside a node."""

    for candidate in candidates:
        text = " ".join(str(candidate or "").split())
        if not text:
            continue
        # A leading clause is often the name: "Four efficiency-vs-fidelity
        # strategies for X" -> take up to the first separator before giving up.
        for piece in (text, text.split(":")[0], text.split(",")[0]):
            piece = piece.strip()
            if 0 < len(piece) <= HUB_LABEL_MAX_CHARS:
                return piece
    return ""


def _slug(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return cleaned[:40] or fallback


def _labels(nodes: Sequence[Any]) -> list[str]:
    return [str(node).strip() for node in nodes if str(node).strip()]


def _split_sides(labels: Sequence[str]) -> tuple[str, list[str], str, list[str]] | None:
    """Split prefixed labels into two named sides, or None if they do not."""

    sides: list[tuple[str, str]] = []
    for label in labels:
        match = _SIDE_RE.match(label)
        if not match:
            return None
        sides.append((match.group("side").strip(), match.group("rest").strip()))
    names: list[str] = []
    for name, _ in sides:
        if name not in names:
            names.append(name)
    if len(names) != 2:
        return None
    left, right = names
    return (
        left,
        [text for name, text in sides if name == left],
        right,
        [text for name, text in sides if name == right],
    )


def _split_branches(labels: Sequence[str]) -> list[tuple[str, list[str]]] | None:
    """Group prefixed labels into 2+ named branches, in first-seen order.

    The branch convention is the comparison convention with one more side
    allowed: ``"Easy: exit cheaply"``, ``"Hard: escalate"``, ``"Hard: then
    reflect"``. Returns None when any label lacks a prefix or fewer than two
    branches result -- a fork with one prong is a pipeline.
    """

    grouped: dict[str, list[str]] = {}
    for label in labels:
        match = _SIDE_RE.match(label)
        if not match:
            return None
        grouped.setdefault(match.group("side").strip(), []).append(match.group("rest").strip())
    if len(grouped) < 2:
        return None
    return list(grouped.items())


# Radius of a region, as a share of the plot: four corner regions fit without touching.
_REGION_EXTENT = 0.2


def _axis_title(axis: Mapping[str, str]) -> str:
    """One title for an axis that a sketch gave as its two ends: ``low -> high``."""

    return f"{axis['low']} \u2192 {axis['high']}"


def _parse_quadrant(labels: Sequence[str]) -> dict[str, Any] | None:
    """Axes plus corner-placed labels, or None if the sketch is not one.

    Needs both axes and at least one corner label; anything less is a list
    wearing a quadrant's name.
    """

    axes: dict[str, dict[str, str]] = {}
    points: list[dict[str, Any]] = []
    for label in labels:
        axis = _AXIS_RE.match(label)
        if axis:
            axes[axis.group("axis").lower()] = {
                "low": axis.group("low").strip(),
                "high": axis.group("high").strip(),
            }
            continue
        corner = _CORNER_RE.match(label)
        if not corner:
            return None
        key = corner.group("corner").lower()
        x, y = _CORNER_XY[key]
        text = corner.group("label")
        points.append({"id": _slug(text, f"p{len(points)}"), "label": text, "x": x, "y": y})
    if "x" not in axes or "y" not in axes or not points:
        return None
    return {"x_axis": axes["x"], "y_axis": axes["y"], "points": points}


def diagram_to_spec(
    sketch: Mapping[str, Any],
    *,
    figure_id: str,
    slide_id: str,
    title: str,
    claim: str,
    takeaway: str,
) -> dict[str, Any]:
    """Build a compiler spec from a ``{kind, caption, nodes}`` sketch."""

    labels = _labels(sketch.get("nodes") or [])
    if not labels:
        raise ValueError(f"diagram sketch for {slide_id!r} has no nodes")
    kind = str(sketch.get("kind") or "").strip().lower()
    archetype = DIAGRAM_KIND_TO_ARCHETYPE.get(kind, DEFAULT_ARCHETYPE)
    caption = str(sketch.get("caption") or "").strip()

    if archetype == "contrast_pair":
        split = _split_sides(labels)
        if split is None:
            # A comparison whose labels do not name two sides is just a list,
            # and forcing it into two panels would invent the grouping.
            archetype = "ranked_list"
        else:
            left_name, left_items, right_name, right_items = split
            content: dict[str, Any] = {
                "left": {"id": _slug(left_name, "left"), "title": left_name, "items": left_items},
                "right": {"id": _slug(right_name, "right"), "title": right_name, "items": right_items},
            }

    if archetype == "branching_paths":
        source_label = labels[0]
        branches = _split_branches(labels[1:])
        if branches is None:
            # A fork whose prongs are not named is a sequence, and the sketch
            # said so; drawing a branch would invent where it splits.
            archetype = "flow"
        else:
            content = {
                "source": {"id": _slug(source_label, "source"), "label": source_label, "kind": "input"},
                "branches": [
                    {
                        "id": _slug(name, f"b{index}"),
                        "label": name,
                        "steps": [
                            {
                                "id": _slug(step, f"b{index}s{step_index}"),
                                "label": step,
                                "kind": "output" if step_index == len(steps) - 1 else "process",
                            }
                            for step_index, step in enumerate(steps)
                        ],
                    }
                    for index, (name, steps) in enumerate(branches)
                ],
            }
    elif archetype == "loop":
        content = {
            "nodes": [
                {"id": _slug(label, f"n{index}"), "label": label, "kind": "process"}
                for index, label in enumerate(labels)
            ]
        }
        centre = _short_label(sketch.get("centre") or "", "")
        if centre:
            content["centre_label"] = centre
    elif archetype == "quadrant_map":
        parsed = _parse_quadrant(labels)
        if parsed is None:
            archetype = "ranked_list"
        else:
            # A sketch places its labels by corner: a qualitative placement, with no
            # measured row behind any position. quadrant_map refuses that (every point
            # needs a ledger row, 8351554409) and names the replacement, which is also
            # what selection.demote picks: labelled regions over two named axes.
            archetype = "landscape_regions"
            content = {
                "x_axis": {"label": _axis_title(parsed["x_axis"])},
                "y_axis": {"label": _axis_title(parsed["y_axis"])},
                "regions": [
                    {"id": point["id"], "label": point["label"], "x": point["x"], "y": point["y"],
                     "extent": _REGION_EXTENT}
                    for point in parsed["points"]
                ],
            }

    if archetype == "flow":
        nodes = [
            {"id": _slug(label, f"n{index}"), "label": label}
            for index, label in enumerate(labels)
        ]
        content = {
            "nodes": nodes,
            "edges": [
                {"from": nodes[index]["id"], "to": nodes[index + 1]["id"]}
                for index in range(len(nodes) - 1)
            ],
        }
    elif archetype == "layered_stack":
        content = {
            "layers": [
                {"id": _slug(label, f"l{index}"), "label": label}
                for index, label in enumerate(labels)
            ]
        }
    elif archetype == "hub_spoke":
        # A sketch may name its own hub ("hub": "Trustworthy deployment"). The
        # fallbacks trim the title or caption, which is right only when one of
        # them has a short leading clause.
        hub_label = _short_label(str(sketch.get("hub") or ""), title, caption)
        if not hub_label:
            # Nothing short enough to name the centre. A hub with a sentence in
            # it is unreadable, and inventing a shorter name would be writing
            # content the sketch does not contain -- so this is a list.
            archetype = "ranked_list"
        else:
            # Spokes feed the hub unless the sketch says the hub produces them.
            # A matrix/grid sketch lists the considerations behind one decision,
            # so outward arrows read backwards: "five attack channels surrounding
            # one defended system" made the system look like the attacker (CVPR
            # 2026), and a "Ship-or-hold scorecard" or "Decision-grade rank"
            # seemed to produce its own audits (ICML 2026, three vision-judge
            # blockers).
            direction = str(sketch.get("spoke_direction") or "to_hub")
            direction = direction if direction in ("from_hub", "to_hub") else "to_hub"
            content = {
                "hub": {"id": _slug(hub_label, "hub"), "label": hub_label},
                "spokes": [
                    {"id": _slug(label, f"s{index}"), "label": label, "kind": "process", "direction": direction}
                    for index, label in enumerate(labels)
                ],
            }
    if archetype == "ranked_list":
        content = {
            "items": [
                {"id": _slug(label, f"i{index}"), "label": label}
                for index, label in enumerate(labels)
            ]
        }

    return {
        "spec_version": "figure-spec/1",
        "figure_id": figure_id,
        "slide_id": slide_id,
        "archetype": archetype,
        "grounding": {"kind": "schematic"},
        # Never a hard character truncation: the geometry gate refuses a line
        # that reads as cut off, and it is right to -- a caption sliced at 60
        # characters ends mid-sentence. Trim on word boundaries instead.
        "title": _headline(title, TITLE_MAX_WORDS),
        "claim": claim or caption or title,
        "takeaway": _headline(takeaway or caption or title),
        "content": content,
    }
