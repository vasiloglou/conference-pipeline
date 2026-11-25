"""Build a spec that sits exactly on an archetype's published budget.

The compiler publishes two numbers per archetype -- how many primary items a
figure may carry and how long each label may be -- and the authoring prompt
hands both to the model as hard limits. Nothing checked that a figure built to
those limits could actually be drawn, and for sixteen of thirty-five it could
not: a six-frame strip and a five-region landscape sat *exactly* at their
stated maximum and were refused, after the paid authoring call had already
been spent. A budget the layout cannot honour is not a budget, it is a bill
the author pays twice.

This module grows each worked example to its declared budget so a test can
assert the contract holds.
"""

from __future__ import annotations

import copy
from typing import Any

from conference_pipeline.figures.examples import EXAMPLES
from conference_pipeline.figures.spec import ARCHETYPES

# The keys `_count_primary` reads, in its order, so the case grows the list the
# budget is actually measured against.
PRIMARY_ORDER = ("nodes", "rows", "bars", "series", "items", "points", "panels", "frames",
                 "curves", "layers", "events", "parts", "modalities", "primitives",
                 "children", "spokes", "actors", "regions", "sets", "axes", "trajectories",
                 "branches")
# Neutral filler: nothing that reads as a pipeline stage or an alternative, so
# the structure lint has no opinion about a spec that exists to test geometry.
# It leads with a long word on purpose. A character budget measured with short
# words is optimistic for any archetype whose columns are narrow: a six-frame
# strip gives each caption about 70px, and one 13-letter word is 124px, so the
# figure refuses on a label well inside its published length.
WORDS = ("independently representation reconstruction measured region sample outline "
         "surface margin window pattern signal texture boundary contrast profile "
         "density gradient").split()
# Strings that are references, vocabulary or units rather than drawn labels.
NOT_A_LABEL = frozenset({
    "id", "kind", "glyph", "shape", "depicts", "from", "to", "row", "col",
    "asset", "unit", "source", "value", "archetype", "ref", "figure", "exit_from",
    "time_label", "overlap_label", "layout",
})
GOVERNED = ("label", "caption", "title", "text", "name")
POSITION = frozenset({"x", "y", "at", "position", "share", "value", "before", "after"})


def filler(chars: int, seed: int) -> str:
    """A label of at most ``chars`` characters, built from whole words."""

    out: list[str] = []
    total = 0
    index = seed
    while True:
        word = WORDS[index % len(WORDS)]
        if total + len(word) + (1 if out else 0) > chars:
            break
        out.append(word)
        total += len(word) + (1 if len(out) > 1 else 0)
        index += 1
    return " ".join(out) or WORDS[seed % len(WORDS)][:max(1, chars)]


def primary_key(content: dict[str, Any]) -> str | None:
    return next((key for key in PRIMARY_ORDER if isinstance(content.get(key), list)), None)


def at_budget(archetype: str, count: int | None = None, chars: int | None = None):
    """The worked example, grown to ``count`` items each labelled ``chars`` long.

    Returns ``None`` for the archetypes whose structure is fixed -- a tug of
    war has two sides and an evidence crop has one image -- where there is no
    list to grow and the budget is satisfied by construction.
    """

    arch = ARCHETYPES[archetype]
    count = arch.max_primary if count is None else count
    chars = arch.max_label_chars if chars is None else chars
    payload = copy.deepcopy(EXAMPLES[archetype])
    content = payload["content"]
    key = primary_key(content)
    if key is None or not content[key]:
        return None

    items = content[key]
    base = len(items)
    # An archetype that draws its nodes INSIDE containers is used that way: a
    # nine-module architecture is three systems of three, never one system and
    # a loose column of seven. Grown nodes join the existing groups in turn, so
    # the case measures the layout's budget rather than a column no author
    # would write.
    groups = [group for group in (content.get("groups") or []) if isinstance(group, dict)]
    while len(items) < count:
        clone = copy.deepcopy(items[len(items) % base])
        if isinstance(clone, dict) and "id" in clone:
            clone["id"] = f"{key[:3]}{len(items)}"
            if key == "nodes" and groups:
                members = groups[(len(items) - base) % len(groups)].setdefault("members", [])
                if isinstance(members, list):
                    members.append(clone["id"])
        items.append(clone)
    del items[count:]
    # A data-grounded point list cites one measured row per point.
    grounding = payload.get("grounding") or {}
    if key == "points" and grounding.get("kind") in ("data", "hybrid"):
        grounding["ledger_rows"] = [f"nc-{i:04d}" for i in range(len(items))]
    kept = {item.get("id") for item in items if isinstance(item, dict)}
    for group in groups:
        members = group.get("members")
        if isinstance(members, list):
            group["members"] = [member for member in members if member in kept]
    if groups:
        # A system with no modules left in it is not a spec anyone writes.
        content["groups"] = [group for group in groups if group.get("members")]

    # References into the primary list go stale when it is trimmed.
    for ref_key in ("links", "highlight"):
        values = content.get(ref_key)
        if isinstance(values, list):
            content[ref_key] = [
                value for value in values
                if not (isinstance(value, int) and value >= count)
                and not (isinstance(value, dict) and isinstance(value.get("from"), int)
                         and (value["from"] >= count or int(value.get("to", 0)) >= count))
            ]

    # Spread positional fields, so cloned items do not land on one another: an
    # overlap the case invents is not a defect in the layout.
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        for field, value in list(item.items()):
            if field in POSITION and isinstance(value, (int, float)) and not isinstance(value, bool):
                item[field] = (round((index + 0.5) / len(items), 3) if 0.0 <= float(value) <= 1.0
                               else round(float(value) * (0.45 + 0.55 * (len(items) - index) / len(items)), 2))

    edges = content.get("edges")
    if isinstance(edges, list) and key == "nodes":
        ids = [node.get("id") for node in items if isinstance(node, dict)]
        content["edges"] = [edge for edge in edges
                            if edge.get("from") in ids and edge.get("to") in ids]
        present = {(edge.get("from"), edge.get("to")) for edge in content["edges"]}
        for source, target in zip(ids, ids[1:]):
            if (source, target) not in present:
                content["edges"].append({"from": source, "to": target})

    counter = [0]

    def relabel(node: Any) -> None:
        if isinstance(node, dict):
            for field, value in node.items():
                if field in ("spec", "grounding"):
                    continue  # a composite panel carries a whole nested spec
                if field in GOVERNED and isinstance(value, str) and value:
                    counter[0] += 1
                    node[field] = filler(chars, counter[0])
                elif isinstance(value, str) and value and field not in NOT_A_LABEL:
                    # Held short, so an ungoverned string is never what binds.
                    counter[0] += 1
                    node[field] = filler(16, counter[0])
                else:
                    relabel(value)
        elif isinstance(node, list):
            for item in node:
                relabel(item)

    relabel(content)

    if key == "rows":
        cols = [col for col in (content.get("cols") or []) if isinstance(col, dict)]
        template = (content.get("cells") or [{}])[0]
        if cols and isinstance(content.get("cells"), list):
            content["cells"] = [
                {**copy.deepcopy(template),
                 "row": row.get("id") or row.get("label"),
                 "col": col.get("id") or col.get("label")}
                for row in items for col in cols
            ]
    return payload
