"""The declarative figure spec: the boundary between the model and the compiler.

The model emits WHAT a figure claims -- entities verbatim, relations, numbers
with ledger rows, emphasis, a takeaway. It may not emit coordinates, sizes,
colours, fonts, marker geometry or a renderer choice; those belong to the
compiler. Anything outside this contract is a spec error, not a layout hint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from conference_pipeline.figures.errors import FigureSpecError

SPEC_VERSION = "figure-spec/1"

FAMILY_TOPOLOGY = "topology"
FAMILY_COMPARISON = "comparison"
FAMILY_QUANTITATIVE = "quantitative"
FAMILY_SCHEMATIC = "schematic"
FAMILY_VISUAL = "visual"
FAMILY_COMPOSITE = "composite"
# Family G: the slide's own words, typeset. Not a picture of anything, so it is
# never chosen by selection or suggested by the critic; it exists for the
# operator outcome "replace the figure with the slide's text layout"
# (docs/operations/figure-fidelity-policy.md, item 9) and is left out of the
# per-deck variety quotas, which measure how varied the PICTURES are.
FAMILY_TEXT = "text"
QUOTA_EXEMPT_FAMILIES = frozenset({FAMILY_TEXT})

GROUNDING_KINDS = ("data", "schematic", "hybrid", "evidence_crop")
SIZE_CLASSES = ("full", "half", "third")
READING_DIRECTIONS = ("lr", "tb")

NODE_KINDS = ("input", "process", "model", "data", "decision", "output", "frozen", "adapted", "metric", "merge")
EDGE_KINDS = ("flow", "feedback", "contrast", "dashed", "parallel")
# "not_reported" is the honest cell: no assigned source states the property
# for that row. Without it the author filled every cell by inference and the
# critic flagged the invented values (ICML 2026 decks 01/02, 2026-10-01).
CELL_VALUES = ("yes", "no", "partial", "not_reported")
# Which way a hub_spoke arrow points. The archetype drew every spoke as
# "hub produces this" and had no way to say the opposite, so a claim like
# "three levers converge on one tension" came out with the tension radiating
# outward INTO its own causes -- backwards, and invisible to every geometry
# check because the arrow was perfectly well drawn.
SPOKE_DIRECTIONS = ("from_hub", "to_hub")

# Shapes a schematic curve may take. Closed vocabulary: a curve the compiler
# cannot name is a curve the reader cannot trust.
CURVE_SHAPES = (
    "monotone_up", "monotone_down", "saturating", "decaying",
    "u_shape", "inverted_u", "knee", "threshold", "oscillating", "flat", "linear_up",
)

# One vocabulary, defined where the glyphs are actually drawn. It used to be
# written out twice -- here and in `glyphs.py` -- with nothing checking that the
# two agreed, so a glyph could be validated and then not exist.
from conference_pipeline.figures.glyphs import (  # noqa: E402
    GLYPH_MODIFIERS,
    GLYPH_NAMES as GLYPHS,
)


# Every top-level `content` key each archetype's layout reads. A key outside
# this list is content the compiler will never draw: not an error the geometry
# gate can see, because nothing is wrong on the canvas, but a silent
# disagreement between the spec's author and the layout -- and reliably a sign
# that the wrong archetype was chosen. Named in an error rather than ignored.
CONTENT_KEYS: dict[str, frozenset[str]] = {
    # topology
    "flow": frozenset({"nodes", "edges"}),
    "loop": frozenset({"nodes", "edges", "centre_label", "output"}),
    "nested_architecture": frozenset({"nodes", "edges", "groups"}),
    "layered_stack": frozenset({"layers", "rail_label"}),
    "branching_paths": frozenset({"source", "branches"}),
    "hierarchy_tree": frozenset({"root", "children"}),
    "decision_tree": frozenset({"root", "children"}),
    "swimlane_sequence": frozenset({"actors", "steps"}),
    "hub_spoke": frozenset({"hub", "spokes"}),
    # comparison
    "feature_matrix": frozenset({"rows", "cols", "cells"}),
    "contrast_pair": frozenset({"left", "right"}),
    "quadrant_map": frozenset({"x_axis", "y_axis", "points", "quadrant_labels"}),
    "ranked_list": frozenset({"items"}),
    "venn_overlap": frozenset({"sets", "overlap_label"}),
    # quantitative
    "bar_compare": frozenset({"metric", "bars"}),
    "paired_delta": frozenset({"metric", "series"}),
    "stat_callout": frozenset({"items"}),
    "line_curve": frozenset({"x", "y", "series", "annotations", "baseline"}),
    "scatter_frontier": frozenset({"x", "y", "points", "frontier"}),
    "radar_profile": frozenset({"axes", "series"}),
    "distribution_breakdown": frozenset({"total_label", "parts"}),
    "heatmap_grid": frozenset({"rows", "cols", "cells", "metric"}),
    "timeline_data": frozenset({"events"}),
    "magnitude_ladder": frozenset({"metric", "bars"}),
    # schematic
    "schematic_curve": frozenset({"x", "y", "curves", "annotations", "regimes"}),
    "trajectory_on_manifold": frozenset({"manifold", "trajectories", "vectors"}),
    "landscape_regions": frozenset({"x_axis", "y_axis", "regions"}),
    "tug_of_war": frozenset({"left", "right", "outcome", "note"}),
    # visual
    "image_transform_strip": frozenset({"panels"}),
    "video_frame_strip": frozenset({"frames", "links", "highlight", "time_label"}),
    "spatial_3d_schematic": frozenset({"primitives", "edges"}),
    "token_grid_overlay": frozenset({"grid", "notes", "legend"}),
    "modality_fusion": frozenset({"modalities", "fusion"}),
    "evidence_crop": frozenset({"crop"}),
    # composite
    "composite": frozenset({"panels", "layout"}),
    # text
    "key_points": frozenset({"points"}),
}


@dataclass(frozen=True)
class Archetype:
    """One figure form: its family, its payload contract and its escape hatch."""

    name: str
    family: str
    summary: str
    required_content: tuple[str, ...]
    grounding_kinds: tuple[str, ...]
    demotes_to: str = ""
    max_primary: int = 12          # nodes / rows / series, whichever the family counts
    # How long a label may be AT A GIVEN item count, as ascending
    # (max_count, max_chars) steps. One ceiling cannot say the true thing:
    # measured at the item budget it forbids labels that draw perfectly with
    # fewer items, and measured at one item it promises room four items do not
    # leave. Empty means the flat `max_label_chars` at every count.
    label_budget: tuple[tuple[int, int], ...] = ()
    max_label_chars: int = 72
    max_secondary: int = 18        # edges / cells / points


    def label_chars(self, count: int) -> int:
        """The label allowance at ``count`` primary items."""

        for max_count, max_chars in self.label_budget:
            if count <= max_count:
                return max_chars
        return self.max_label_chars

    def budget_line(self) -> str:
        """The budget as one line for the authoring prompt."""

        if not self.label_budget:
            return (f"up to {self.max_primary} items, labels up to "
                    f"{self.max_label_chars} characters")
        parts = [f"{'up to' if i == 0 else 'up to'} {c} items: {n} chars"
                 for i, (c, n) in enumerate(self.label_budget)]
        return (f"up to {self.max_primary} items; the label allowance depends on how "
                f"many you use -- " + "; ".join(parts))


def _a(name, family, summary, required, grounding, demotes_to="", max_primary=12,
       max_label_chars=72, max_secondary=18, label_budget=()) -> Archetype:
    # `max_label_chars` is the allowance at the item budget -- the conservative
    # headline. `label_budget` is the measured curve behind it, so a spec with
    # fewer items is judged against the room it actually has.
    steps = tuple(label_budget)
    return Archetype(
        name=name, family=family, summary=summary, required_content=tuple(required),
        grounding_kinds=tuple(grounding), demotes_to=demotes_to, max_primary=max_primary,
        max_label_chars=steps[-1][1] if steps else max_label_chars,
        max_secondary=max_secondary, label_budget=steps,
    )


_STRUCTURAL = ("schematic", "hybrid", "data")
_DATA_ONLY = ("data", "hybrid")

ARCHETYPES: dict[str, Archetype] = {
    # -- Family A: topology --------------------------------------------------
    "flow": _a("flow", FAMILY_TOPOLOGY, "A directed process with stages.",
               ("nodes", "edges"), _STRUCTURAL, max_primary=6),
    # Re-measured once arrows had to show a shaft (figure-compiler/1.1.0): the
    # earlier curves were honoured only because a 12-18px gap between linked
    # boxes drew a head with no line under it and nothing measured that.
    "loop": _a("loop", FAMILY_TOPOLOGY, "A cyclic process that returns to its start.",
               ("nodes",), _STRUCTURAL, max_primary=7,
                      label_budget=((5, 72), (6, 60), (7, 26),)),
    # Two containers, members stacked with room for the arrows between them:
    # four two-line modules fill a column. Nine modules need three systems,
    # which an author may still write -- the budget is advisory at compile time
    # and a spec that draws, draws.
    "nested_architecture": _a("nested_architecture", FAMILY_TOPOLOGY,
                              "Modules contained inside systems; frozen vs adapted parts.",
                              ("nodes", "groups"), _STRUCTURAL, max_primary=6,
                      label_budget=((4, 72), (6, 48),)),
    "layered_stack": _a("layered_stack", FAMILY_TOPOLOGY,
                        "Ordered layers with a quantity that changes over depth.",
                        ("layers",), _STRUCTURAL, max_primary=8),
    "branching_paths": _a("branching_paths", FAMILY_TOPOLOGY,
                          "Two alternative paths from one input, one emphasised.",
                          ("source", "branches"), _STRUCTURAL, max_primary=3),
    "hierarchy_tree": _a("hierarchy_tree", FAMILY_TOPOLOGY, "A taxonomy or decomposition.",
                         ("root", "children"), _STRUCTURAL, max_primary=6,
                      label_budget=((2, 72), (3, 60), (6, 40),)),
    "swimlane_sequence": _a("swimlane_sequence", FAMILY_TOPOLOGY,
                            "Actors exchanging steps over time.",
                            ("actors", "steps"), _STRUCTURAL, max_primary=5, max_secondary=9,
                      label_budget=((2, 60), (3, 32), (4, 26), (5, 20),)),
    "hub_spoke": _a("hub_spoke", FAMILY_TOPOLOGY, "One hub with anchored satellites.",
                    ("hub", "spokes"), _STRUCTURAL, max_primary=7,
                    label_budget=((7, 60),)),
    # -- Family B: comparison ------------------------------------------------
    "feature_matrix": _a("feature_matrix", FAMILY_COMPARISON,
                         "Methods against properties, with yes/no/partial cells.",
                         ("rows", "cols", "cells"), _STRUCTURAL, max_primary=6, max_secondary=30,
                      label_budget=((4, 72), (6, 60),)),
    "contrast_pair": _a("contrast_pair", FAMILY_COMPARISON, "A vs B on aligned attributes.",
                        ("left", "right"), _STRUCTURAL, max_primary=6),
    # Data-only since 2026-10-01: a point's position reads as a measurement
    # whatever a 'basis' says, so every point is placed from a ledger row or
    # the figure is landscape_regions (where a method applies, no positions).
    "quadrant_map": _a("quadrant_map", FAMILY_COMPARISON,
                       "Methods positioned on two axes, each point placed from a measured row.",
                       ("x_axis", "y_axis", "points"), _DATA_ONLY, max_primary=8,
                      label_budget=((4, 72), (5, 60), (6, 48), (8, 40),)),
    "ranked_list": _a("ranked_list", FAMILY_COMPARISON, "An ordering by one attribute.",
                      ("items",), _STRUCTURAL, max_primary=8),
    "decision_tree": _a("decision_tree", FAMILY_COMPARISON, "Which method when: routing by question.",
                        ("root", "children"), _STRUCTURAL, max_primary=7,
                      label_budget=((4, 72), (5, 60), (7, 40),)),
    "venn_overlap": _a("venn_overlap", FAMILY_COMPARISON, "Set relationships between two or three sets.",
                       ("sets",), _STRUCTURAL, max_primary=3,
                       label_budget=((2, 60), (3, 32),)),
    # -- Family C: quantitative ----------------------------------------------
    "bar_compare": _a("bar_compare", FAMILY_QUANTITATIVE, "One metric across methods or conditions.",
                      ("metric", "bars"), _DATA_ONLY, demotes_to="stat_callout", max_primary=6),
    "paired_delta": _a("paired_delta", FAMILY_QUANTITATIVE, "Before to after per method, with deltas.",
                       ("metric", "series"), _DATA_ONLY, demotes_to="stat_callout", max_primary=6),
    # With no number to call out, the honest shape is the scorecard: what was
    # evaluated against what, with yes/no/partial cells and no invented value.
    "stat_callout": _a("stat_callout", FAMILY_QUANTITATIVE, "One to three headline numbers with context.",
                       ("items",), _DATA_ONLY, demotes_to="feature_matrix", max_primary=3),
    "line_curve": _a("line_curve", FAMILY_QUANTITATIVE, "A measured metric against a parameter.",
                     ("x", "y", "series"), _DATA_ONLY, demotes_to="schematic_curve", max_primary=2, max_label_chars=40),
    "scatter_frontier": _a("scatter_frontier", FAMILY_QUANTITATIVE,
                           "Two metrics per method, with an optional Pareto frontier.",
                           ("x", "y", "points"), _DATA_ONLY, demotes_to="quadrant_map",
                           max_primary=6),
    "radar_profile": _a("radar_profile", FAMILY_QUANTITATIVE, "A multi-axis profile.",
                        ("axes", "series"), _DATA_ONLY, demotes_to="hierarchy_tree", max_primary=8,
                      label_budget=((4, 48), (8, 32),)),
    "distribution_breakdown": _a("distribution_breakdown", FAMILY_QUANTITATIVE, "Shares of a whole.",
                                 ("total_label", "parts"), _DATA_ONLY, demotes_to="stat_callout", max_primary=6,
                      label_budget=((3, 72), (4, 60), (6, 48),)),
    "heatmap_grid": _a("heatmap_grid", FAMILY_QUANTITATIVE, "A numeric matrix over two categories.",
                       ("rows", "cols", "cells"), _DATA_ONLY, demotes_to="feature_matrix",
                       max_primary=6, max_secondary=36),
    "timeline_data": _a("timeline_data", FAMILY_QUANTITATIVE, "Dated events, versions or lineage.",
                        ("events",), _STRUCTURAL, demotes_to="flow", max_primary=7),
    "magnitude_ladder": _a("magnitude_ladder", FAMILY_QUANTITATIVE, "Orders of magnitude on a log scale.",
                           ("metric", "bars"), _DATA_ONLY, demotes_to="stat_callout", max_primary=7),
    # -- Family D: schematic shapes ------------------------------------------
    "schematic_curve": _a("schematic_curve", FAMILY_SCHEMATIC,
                          "The qualitative shape of a behaviour, with named regimes.",
                          ("x", "y", "curves"), ("schematic", "hybrid"), max_primary=2),
    "trajectory_on_manifold": _a("trajectory_on_manifold", FAMILY_SCHEMATIC,
                                 "Paths through a space, on or off a surface.",
                                 ("manifold", "trajectories"), ("schematic", "hybrid"), max_primary=3),
    "landscape_regions": _a("landscape_regions", FAMILY_SCHEMATIC,
                            "Where a method applies, as regions over two axes.",
                            ("x_axis", "y_axis", "regions"), ("schematic", "hybrid"), max_primary=5,
                      label_budget=((2, 72), (3, 32), (4, 26), (5, 20),)),
    "tug_of_war": _a("tug_of_war", FAMILY_SCHEMATIC, "Two forces competing for one outcome.",
                     ("left", "right"), ("schematic", "hybrid"), max_primary=2),
    # -- Family E: visual-domain ---------------------------------------------
    "image_transform_strip": _a("image_transform_strip", FAMILY_VISUAL,
                                "Input image, operation, output image.",
                                ("panels",), _STRUCTURAL, max_primary=5),
    "video_frame_strip": _a("video_frame_strip", FAMILY_VISUAL,
                            "A temporal claim over consecutive frames.",
                            ("frames",), _STRUCTURAL, max_primary=6),
    "spatial_3d_schematic": _a("spatial_3d_schematic", FAMILY_VISUAL,
                               "Scenes, cameras, splats and actions in 3-D.",
                               ("primitives",), _STRUCTURAL, max_primary=7,
                      label_budget=((5, 72), (7, 60),)),
    "token_grid_overlay": _a("token_grid_overlay", FAMILY_VISUAL,
                             "Patches or tokens kept, pruned or attended.",
                             ("grid",), _STRUCTURAL, max_primary=4),
    "modality_fusion": _a("modality_fusion", FAMILY_VISUAL, "Modalities converging, some missing.",
                          ("modalities", "fusion"), _STRUCTURAL, max_primary=4,
                      label_budget=((1, 72), (3, 48), (4, 26),)),
    "evidence_crop": _a("evidence_crop", FAMILY_VISUAL, "A licensed crop of a published figure.",
                        ("crop",), ("evidence_crop",), max_primary=1),
    # -- Family F: composite --------------------------------------------------
    "composite": _a("composite", FAMILY_COMPOSITE, "Two or three panels under one header.",
                    ("panels",), _STRUCTURAL, max_primary=3),
    # -- Family G: text -------------------------------------------------------
    "key_points": _a("key_points", FAMILY_TEXT,
                     "The slide's own visible text and claim, typeset; nothing added.",
                     ("points",), ("schematic",), max_primary=6,
                     label_budget=((4, 140), (6, 96),)),
}

FAMILY_OF = {name: arch.family for name, arch in ARCHETYPES.items()}


@dataclass(frozen=True)
class Grounding:
    kind: str
    ledger_rows: tuple[str, ...] = ()
    license: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.ledger_rows:
            payload["ledger_rows"] = list(self.ledger_rows)
        if self.license:
            payload["license"] = self.license
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class FigureSpec:
    """A validated figure spec."""

    figure_id: str
    slide_id: str
    archetype: str
    claim: str
    takeaway: str
    grounding: Grounding
    content: dict[str, Any]
    title: str = ""
    # Editorial section the slide belongs to (blueprint ``section_id``). Carried
    # so the per-deck variety quota can ask whether each section shows a number,
    # not just whether the deck as a whole does.
    section: str = ""
    emphasis: dict[str, Any] = field(default_factory=dict)
    reading_direction: str = "lr"
    size_class: str = "full"
    constraints: dict[str, Any] = field(default_factory=dict)
    required_entities: tuple[str, ...] = ()
    spec_version: str = SPEC_VERSION
    synthesis_label: str = ""

    @property
    def family(self) -> str:
        return FAMILY_OF[self.archetype]

    @property
    def archetype_def(self) -> Archetype:
        return ARCHETYPES[self.archetype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "figure_id": self.figure_id,
            "slide_id": self.slide_id,
            "archetype": self.archetype,
            "claim": self.claim,
            "takeaway": self.takeaway,
            **({"synthesis_label": self.synthesis_label} if self.synthesis_label else {}),
            "title": self.title,
            "section": self.section,
            "grounding": self.grounding.to_dict(),
            "emphasis": dict(self.emphasis),
            "reading_direction": self.reading_direction,
            "size_class": self.size_class,
            "required_entities": list(self.required_entities),
            "constraints": dict(self.constraints),
            "content": self.content,
        }

    def spec_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -- validation ---------------------------------------------------------------

FORBIDDEN_CONTENT_KEYS = frozenset(
    {"x_px", "y_px", "width", "height", "color", "colour", "fill", "stroke",
     "font", "font_size", "coordinates", "svg", "path", "marker", "renderer", "engine"}
)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _walk_forbidden(node: Any, path: str, errors: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in FORBIDDEN_CONTENT_KEYS:
                errors.append(
                    f"{path}.{key}: the spec may not carry presentation detail "
                    f"('{key}'); the compiler owns layout, colour and typography"
                )
            _walk_forbidden(value, f"{path}.{key}", errors)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk_forbidden(item, f"{path}[{index}]", errors)


def _count_primary(archetype: str, content: dict[str, Any]) -> int:
    for key in ("nodes", "rows", "bars", "series", "items", "points", "panels", "frames",
                "curves", "layers", "events", "parts", "modalities", "primitives",
                "children", "spokes", "actors", "regions", "sets", "axes", "trajectories",
                "branches"):
        value = content.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _numbers_in_content(content: Any) -> list[float]:
    found: list[float] = []
    if isinstance(content, dict):
        for key, value in content.items():
            if key in {"value", "before", "after", "y", "x", "share", "count"} and isinstance(value, (int, float)):
                found.append(float(value))
            else:
                found.extend(_numbers_in_content(value))
    elif isinstance(content, list):
        for item in content:
            found.extend(_numbers_in_content(item))
    return found


def budget_findings(spec: FigureSpec) -> list[str]:
    """Where a spec exceeds its archetype's published item and label budgets.

    These are a floor, not a boundary. Each one is measured by growing a single
    worked example, and the worst case across the whole count range is the
    number published -- so a spec above the budget may still draw, because its
    own topology is kinder than the shape that set the limit. Three of the
    forty-nine CVPR specs are exactly that.

    So the budget is an error while AUTHORING, where it is worth having and a
    fix costs one cheap call, and an advisory at COMPILE time, where the
    drawing itself is the better test. Refusing a figure that demonstrably
    draws buys nothing, and the geometry gate still catches the ones that do
    not.
    """

    arch = ARCHETYPES.get(spec.archetype)
    if arch is None:
        return []
    findings: list[str] = []
    primary = _count_primary(spec.archetype, spec.content)
    if primary > arch.max_primary:
        findings.append(
            f"{primary} primary items exceeds the budget of {arch.max_primary} for "
            f"'{spec.archetype}'; split into a composite or drop the weakest items"
        )
    edges = spec.content.get("edges")
    if isinstance(edges, list) and len(edges) > arch.max_secondary:
        findings.append(f"{len(edges)} edges exceeds the budget of {arch.max_secondary}")

    allowance = arch.label_chars(primary)
    for label in _collect_labels(spec.content):
        if len(label) > allowance:
            findings.append(
                f"label '{label[:40]}...' is {len(label)} characters; at {primary} items "
                f"'{spec.archetype}' allows {allowance}. Shorten the label, carry "
                "fewer items, or do both -- the figure will not shrink to fit."
            )
    return findings


def validate_spec(spec: FigureSpec, *, mode: str = "authoring") -> list[str]:
    """Every problem with ``spec``, as human-readable repair instructions.

    ``mode`` "compile" drops the budget checks, which are advisory once the
    figure is being drawn -- see `budget_findings`. Everything else, including
    referential integrity and the structure lint, refuses in both modes.
    """

    errors: list[str] = []
    if not isinstance(spec.synthesis_label, str) or len(spec.synthesis_label) > 64:
        errors.append("synthesis_label must be a string of at most 64 characters")
    if spec.spec_version != SPEC_VERSION:
        errors.append(f"spec_version must be '{SPEC_VERSION}'")
    if spec.archetype not in ARCHETYPES:
        return [f"archetype '{spec.archetype}' is not one of: {', '.join(sorted(ARCHETYPES))}"]
    arch = ARCHETYPES[spec.archetype]

    if not spec.figure_id.strip():
        errors.append("figure_id is required")
    if not spec.claim.strip():
        errors.append("claim is required: one sentence stating what the figure makes legible")
    if not spec.takeaway.strip():
        # A text layout carries only what the slide says; a takeaway the slide
        # does not state would be the one invented line on it.
        if arch.family != FAMILY_TEXT:
            errors.append("takeaway is required")
    elif len(spec.takeaway.split()) > 12:
        errors.append(f"takeaway must be at most 12 words (got {len(spec.takeaway.split())})")
    if spec.size_class not in SIZE_CLASSES:
        errors.append(f"size_class must be one of {SIZE_CLASSES}")
    if spec.reading_direction not in READING_DIRECTIONS:
        errors.append(f"reading_direction must be one of {READING_DIRECTIONS}")

    # Grounding ---------------------------------------------------------------
    if spec.grounding.kind not in GROUNDING_KINDS:
        errors.append(f"grounding.kind must be one of {GROUNDING_KINDS}")
    elif spec.grounding.kind not in arch.grounding_kinds:
        errors.append(
            f"archetype '{spec.archetype}' accepts grounding {arch.grounding_kinds}, "
            f"not '{spec.grounding.kind}'"
            + (f"; with no measured numbers use '{_schematic_fallback(spec.archetype)}'"
               if _schematic_fallback(spec.archetype) else "")
        )
    if spec.grounding.kind == "data" and not spec.grounding.ledger_rows:
        errors.append(
            "grounding.kind 'data' requires ledger_rows: every plotted value must name a "
            "numeric-claims ledger row. With no measured rows, demote the archetype instead "
            "of drawing illustrative values."
        )
    if spec.grounding.kind == "evidence_crop" and not spec.grounding.license.strip():
        errors.append("evidence_crop requires grounding.license naming an adaptable licence")
    if arch.family == FAMILY_QUANTITATIVE and spec.archetype != "timeline_data":
        if spec.grounding.kind == "schematic":
            errors.append(
                f"'{spec.archetype}' plots measured values and cannot be schematic; "
                f"demote to '{_schematic_fallback(spec.archetype) or 'feature_matrix'}'"
            )
        if _numbers_in_content(spec.content) and not spec.grounding.ledger_rows:
            errors.append("numeric content requires grounding.ledger_rows")

    # Content -----------------------------------------------------------------
    if not isinstance(spec.content, dict):
        return [*errors, "content must be an object"]
    for key in arch.required_content:
        if key not in spec.content or spec.content.get(key) in (None, "", [], {}):
            errors.append(f"content.{key} is required for archetype '{spec.archetype}'")
    _walk_forbidden(spec.content, "content", errors)

    if mode != "compile":
        errors.extend(budget_findings(spec))
    errors.extend(_validate_payload(spec, arch))
    # Structure last: a spec that is not even shaped right should hear about
    # that before it hears that its shape says the wrong thing.
    errors.extend(structure_findings(spec))
    return errors


# Every content key a layout reads as free text the reader is meant to see.
# `_collect_labels` below uses the narrow set for the LENGTH budget; this wider
# set answers a different question -- "what did the spec ask to be drawn?" --
# which the manifest compares against what actually reached the canvas.
LABEL_KEYS = frozenset({
    "label", "title", "text", "caption", "name", "sublabel", "edge_label", "note",
    "via", "when", "delta_label", "comparator", "rail_label", "centre_label",
    "overlap_label", "total_label", "time_label", "outcome", "badge", "low", "high",
})

# Keys whose value is a plain string that names a THING, not text drawn on the
# canvas: skipping them keeps ids and vocabulary words out of the accounting.
_NON_LABEL_KEYS = frozenset({
    "id", "kind", "glyph", "style", "shape", "from", "to", "row", "col", "group",
    "source", "asset", "license", "unit", "scale", "position", "params", "members",
    "highlight", "dim", "kept", "ledger_rows", "on_frontier", "missing", "steps",
    "basis",
})

# Lists whose bare strings ARE drawn text (`contrast_pair` sides, ranked lists).
_BARE_STRING_LABEL_KEYS = frozenset({"items", "bullets", "points_text"})


def content_labels(node: Any) -> list[str]:
    """Every label a spec asks the layout to draw, in document order.

    Deduplicated on the normalised string, because the same word appearing as
    two nodes' labels is one question for the manifest ("did it render?"), not
    two.
    """

    found: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        body = " ".join(str(value).split())
        if body and body.lower() not in seen:
            seen.add(body.lower())
            found.append(body)

    def walk(value: Any, parent: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key)
                if name in _NON_LABEL_KEYS:
                    continue
                if name in LABEL_KEYS and isinstance(item, str):
                    add(item)
                else:
                    walk(item, name)
        elif isinstance(value, list):
            for item in value:
                # Bare strings are drawn text only under the keys that carry
                # them as prose. Everywhere else a bare string in a list is a
                # reference -- `highlight: ["f2"]`, `members: [...]` -- and
                # counting it as a label would demand the compiler render an id.
                if isinstance(item, str):
                    if parent in _BARE_STRING_LABEL_KEYS:
                        add(item)
                else:
                    walk(item, parent)

    walk(node, "")
    return found


def _collect_labels(node: Any) -> list[str]:
    labels: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"label", "title", "text", "caption", "name"} and isinstance(value, str):
                labels.append(value)
            else:
                labels.extend(_collect_labels(value))
    elif isinstance(node, list):
        for item in node:
            labels.extend(_collect_labels(item))
    return labels


# The SHAPE each content key must have. Presence was checked and type was not,
# so a spec could name `grid` as a list and reach the layout, which then failed
# with an AttributeError instead of a message anyone could act on. A live model
# produces this; a fuzz corpus built from the worked examples never does.
CONTENT_SHAPES: dict[str, str] = {
    "nodes": "list", "edges": "list", "groups": "list", "layers": "list",
    "branches": "list", "children": "list", "steps": "list", "spokes": "list",
    "rows": "list", "cols": "list", "cells": "list", "items": "list",
    "points": "list", "bars": "list", "series": "list", "parts": "list",
    "events": "list", "curves": "list", "trajectories": "list", "vectors": "list",
    "regions": "list", "sets": "list", "axes": "list", "annotations": "list",
    "panels": "list", "frames": "list", "primitives": "list", "modalities": "list",
    "notes": "list", "links": "list", "highlight": "list", "quadrant_labels": "list",
    "actors": "list", "stages": "list",
    "root": "dict", "hub": "dict", "source": "dict", "metric": "dict",
    "grid": "dict", "fusion": "dict", "crop": "dict", "left": "dict", "right": "dict",
    "x": "dict", "y": "dict", "x_axis": "dict", "y_axis": "dict", "manifold": "dict",
    "output": "dict", "legend": "dict",
}


def _shape_errors(content: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, expected in CONTENT_SHAPES.items():
        if key not in content:
            continue
        value = content[key]
        if expected == "list" and not isinstance(value, list):
            errors.append(f"content.{key} must be a list, not {type(value).__name__}")
        elif expected == "dict" and not isinstance(value, dict):
            errors.append(f"content.{key} must be an object, not {type(value).__name__}")
    # One level down into `fusion`, the only key whose children have shapes of
    # their own. `output` is the one key with two legal shapes -- a result, or
    # a list of results -- and `alternatives` is a list of mechanisms. Both
    # arrived from live authoring, both slipped past a top-level check, and
    # both were drawn as nothing.
    fusion = content.get("fusion")
    if isinstance(fusion, dict):
        output = fusion.get("output")
        if output is not None and not isinstance(output, dict) and not (
            isinstance(output, list) and all(isinstance(item, dict) for item in output)
        ):
            errors.append("content.fusion.output must be an object {label} or a list of them")
        alternatives = fusion.get("alternatives")
        if alternatives is not None and not (
            isinstance(alternatives, list) and all(isinstance(item, dict) for item in alternatives)
        ):
            errors.append("content.fusion.alternatives must be a list of objects {id, label, output}")
    return errors


def structure_findings(spec: FigureSpec, *, deck_claims: tuple[str, ...] = ()) -> list[str]:
    """Lint findings that refuse a spec, as validator messages.

    Imported lazily: `lint` reads `FigureSpec` from here, and a module-level
    import would close the circle.
    """

    from conference_pipeline.figures.lint import lint_spec

    messages: list[str] = []
    for finding in lint_spec(spec, deck_claims=deck_claims):
        if finding.severity == "warn":
            continue
        detail = finding.message
        if finding.demote_to:
            detail += f"; this content is a {finding.demote_to}"
        if finding.suggested_change:
            detail += f" -- {finding.suggested_change}"
        messages.append(f"{finding.code}: {detail}")
    return messages


def _validate_payload(spec: FigureSpec, arch: Archetype) -> list[str]:
    """Archetype-specific structural rules (referential integrity, vocabularies)."""

    content = spec.content
    errors: list[str] = []
    name = spec.archetype

    # Content under a key this layout never reads is content nobody will ever
    # see. Nothing is wrong on the canvas, so no geometry check can catch it,
    # and until the manifest started accounting for labels nothing downstream
    # could either -- it is also a reliable sign that the wrong archetype was
    # chosen for the claim.
    errors.extend(_shape_errors(content))

    known = CONTENT_KEYS.get(name)
    if known is not None:
        for key in sorted(set(content) - known):
            errors.append(
                f"content.{key} is not read by '{name}'; it would never be drawn. "
                f"Keys this archetype uses: {', '.join(sorted(known))}"
            )

    if name in {"flow", "nested_architecture"} or (name == "loop" and content.get("edges")):
        nodes = content.get("nodes") or []
        ids = {str(node.get("id") or "") for node in nodes if isinstance(node, dict)}
        for node in nodes:
            if not isinstance(node, dict):
                errors.append("content.nodes entries must be objects")
                continue
            if not str(node.get("id") or "").strip():
                errors.append("every node needs an id")
            if not str(node.get("label") or "").strip():
                errors.append(f"node '{node.get('id')}' needs a label")
            kind = str(node.get("kind") or "process")
            if kind not in NODE_KINDS:
                errors.append(f"node kind '{kind}' must be one of {NODE_KINDS}")
        for edge in content.get("edges") or []:
            if not isinstance(edge, dict):
                errors.append("content.edges entries must be objects")
                continue
            for endpoint in ("from", "to"):
                if str(edge.get(endpoint) or "") not in ids:
                    errors.append(f"edge {endpoint} '{edge.get(endpoint)}' is not a node id")
            kind = str(edge.get("kind") or "flow")
            if kind not in EDGE_KINDS:
                errors.append(f"edge kind '{kind}' must be one of {EDGE_KINDS}")
        if name == "flow":
            forward: dict[str, set[str]] = {node_id: set() for node_id in ids}
            for edge in content.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                if str(edge.get("kind") or "flow") in {"feedback", "parallel", "contrast"}:
                    continue
                forward.setdefault(str(edge.get("from") or ""), set()).add(str(edge.get("to") or ""))
            for edge in content.get("edges") or []:
                if not isinstance(edge, dict) or edge.get("kind") != "feedback":
                    continue
                source_id, target_id = str(edge.get("from") or ""), str(edge.get("to") or "")
                if source_id == target_id:
                    errors.append("a feedback edge needs two distinct stages")
                    continue
                seen: set[str] = set()
                pending = [target_id]
                while pending:
                    current = pending.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    pending.extend(forward.get(current, ()))
                if source_id not in seen:
                    errors.append(
                        f"feedback edge '{source_id}->{target_id}' needs a forward path "
                        f"from '{target_id}' to '{source_id}' so the loop closes"
                    )
            for node in nodes:
                if not isinstance(node, dict) or node.get("kind") != "merge":
                    continue
                incoming = sum(
                    isinstance(edge, dict) and str(edge.get("to") or "") == str(node.get("id") or "")
                    and str(edge.get("kind") or "flow") not in {"parallel", "contrast", "feedback"}
                    for edge in content.get("edges") or []
                )
                if incoming < 2:
                    errors.append(f"merge node '{node.get('id')}' needs at least two incoming flow edges")
        if name == "nested_architecture":
            for group in content.get("groups") or []:
                if not isinstance(group, dict):
                    errors.append("content.groups entries must be objects")
                    continue
                for member in _strings(group.get("members")):
                    if member not in ids:
                        errors.append(f"group member '{member}' is not a node id")

    if name == "hub_spoke":
        for spoke in content.get("spokes") or []:
            if not isinstance(spoke, dict):
                errors.append("content.spokes entries must be objects")
                continue
            direction = str(spoke.get("direction") or "from_hub")
            if direction not in SPOKE_DIRECTIONS:
                errors.append(
                    f"spoke direction '{direction}' must be one of {SPOKE_DIRECTIONS}; "
                    "'to_hub' for a spoke that feeds the hub, 'from_hub' (the default) "
                    "for one the hub produces"
                )

    if name == "feature_matrix":
        rows = _strings([r.get("label") if isinstance(r, dict) else r for r in content.get("rows") or []])
        cols = _strings([c.get("label") if isinstance(c, dict) else c for c in content.get("cols") or []])
        for cell in content.get("cells") or []:
            if not isinstance(cell, dict):
                errors.append("content.cells entries must be objects")
                continue
            if str(cell.get("row") or "") not in rows:
                errors.append(f"cell row '{cell.get('row')}' is not a declared row label")
            if str(cell.get("col") or "") not in cols:
                errors.append(f"cell col '{cell.get('col')}' is not a declared column label")
            value = str(cell.get("value") or "")
            if value not in CELL_VALUES and not str(cell.get("text") or "").strip():
                errors.append(f"cell value '{value}' must be one of {CELL_VALUES} or carry text")

    if name == "schematic_curve":
        for curve in content.get("curves") or []:
            if not isinstance(curve, dict):
                errors.append("content.curves entries must be objects")
                continue
            shape = str(curve.get("shape") or "")
            if shape not in CURVE_SHAPES:
                errors.append(f"curve shape '{shape}' must be one of {CURVE_SHAPES}")
        for axis in ("x", "y"):
            spec_axis = content.get(axis)
            if isinstance(spec_axis, dict) and spec_axis.get("ticks"):
                errors.append(
                    f"a schematic curve may not carry numeric ticks on {axis}; "
                    "use line_curve with ledger rows for measured values"
                )

    if name == "key_points":
        # Each point is one line of the slide's visible text, verbatim. Only
        # `text` is read; an id, an emphasis or a sublabel would be the author
        # adding structure the slide never had.
        for point in content.get("points") or []:
            if not isinstance(point, dict):
                errors.append("content.points entries must be objects {text}")
                continue
            if not str(point.get("text") or "").strip():
                errors.append("every key point needs non-empty text")
            extra = sorted(set(point) - {"text"})
            if extra:
                errors.append(f"a key point carries only text, not {', '.join(extra)}")

    if name in {"quadrant_map", "scatter_frontier"}:
        for point in content.get("points") or []:
            if not isinstance(point, dict):
                errors.append("content.points entries must be objects")
                continue
            for axis in ("x", "y"):
                value = point.get(axis)
                if not isinstance(value, (int, float)):
                    errors.append(f"point '{point.get('label')}' needs a numeric {axis}")
                elif name == "quadrant_map" and not 0.0 <= float(value) <= 1.0:
                    errors.append(f"quadrant_map {axis} must be in [0, 1] (ordinal position)")
        if name == "quadrant_map":
            # A point's position is a claim. Positions placed from measured
            # rows need one row per point; a qualitative placement needs the
            # source statement that puts it there (``basis``, never drawn), so
            # the critic can check it. Invented positions were the adversarial
            # critic's most common finding on ICML 2026 decks 01/02.
            points = [p for p in content.get("points") or [] if isinstance(p, dict)]
            rows = len(spec.grounding.ledger_rows)
            if points and rows < len(points):
                errors.append(
                    f"quadrant_map places {len(points)} points but grounding.ledger_rows names "
                    f"{rows} measured row(s); every point needs a row -- cite one per point, "
                    "drop the unmeasured points, or author this as 'landscape_regions'"
                )

    if name in _ONE_SCALE_ARCHETYPES:
        # A bar's LENGTH is the claim, so two bars are only comparable when
        # they measure the same thing. Deck 08's slide 13 plotted a 1.49x
        # speedup beside a 21.1% latency reduction on one scale, under the
        # unit "x / %": the percentage drew fourteen times longer and the
        # figure said the opposite of the slide. A compound unit is the tell.
        unit = str((content.get("metric") or {}).get("unit") or "")
        if any(token in unit for token in ("/", " or ", ",", " and ")):
            errors.append(
                f"'{name}' draws every value on one length scale, so they must share one "
                f"unit; metric.unit {unit!r} names more than one. Plot the values that share "
                "a unit and cite the rest in the takeaway, or use 'stat_callout'"
            )
        units = {
            str(row.get("unit") or "").strip()
            for row in (content.get("bars") or content.get("series") or [])
            if isinstance(row, dict) and str(row.get("unit") or "").strip()
        }
        if len(units) > 1:
            errors.append(
                f"'{name}' draws every value on one length scale, so they must share one "
                f"unit; its rows name {sorted(units)}"
            )

    if name == "stat_callout":
        for item in content.get("items") or []:
            if not isinstance(item, dict):
                errors.append("content.items entries must be objects")
                continue
            if item.get("value") in (None, ""):
                errors.append("every stat_callout item needs a value")
            if not str(item.get("label") or "").strip():
                errors.append("every stat_callout item needs a label")
            if not str(item.get("source") or "").strip():
                errors.append("every stat_callout item needs a source id")

    if name in {"image_transform_strip", "video_frame_strip", "modality_fusion", "token_grid_overlay",
                "spatial_3d_schematic"}:
        for panel in (content.get("panels") or content.get("frames") or content.get("modalities")
                      or content.get("primitives") or []):
            if isinstance(panel, dict):
                glyph = str(panel.get("glyph") or panel.get("kind") or "")
                if glyph and glyph not in GLYPHS and name != "spatial_3d_schematic":
                    errors.append(f"glyph '{glyph}' must be one of {GLYPHS}")
                for modifier in (panel.get("modifiers") or []):
                    if str(modifier) not in GLYPH_MODIFIERS:
                        errors.append(
                            f"modifier '{modifier}' must be one of {GLYPH_MODIFIERS}"
                        )

    if name == "composite":
        panels = content.get("panels") or []
        if not 2 <= len(panels) <= 3:
            errors.append("a composite needs two or three panels")
        for panel in panels:
            if not isinstance(panel, dict) or "spec" not in panel:
                errors.append("every composite panel needs a nested 'spec'")
    return errors


# Archetypes whose bar or segment LENGTH encodes the value, so every value
# they draw has to be in the same unit to be comparable at all.
_ONE_SCALE_ARCHETYPES = frozenset({"bar_compare", "paired_delta", "magnitude_ladder"})


def _schematic_fallback(archetype: str) -> str:
    """The archetype to suggest when a spec says it has no measured numbers.

    Following one `demotes_to` hop is not enough: `bar_compare` demotes to
    `stat_callout`, which needs values of its own, so a slide with nothing to
    cite was told to use an archetype it could not fill either and the author
    bounced between the two until its attempts ran out (deck 05, sl_03). Walk
    until the chain leaves the quantitative family -- the same rule
    `selection.demote` applies when it makes the choice itself.
    """

    seen = {archetype}
    target = ARCHETYPES[archetype].demotes_to if archetype in ARCHETYPES else ""
    while target and target not in seen and FAMILY_OF.get(target) == FAMILY_QUANTITATIVE:
        seen.add(target)
        target = ARCHETYPES[target].demotes_to
    return target


def normalise_emphasis(value: Any) -> dict[str, Any]:
    """The one emphasis shape the layouts read: ``{"highlight": [ids], "dim": [ids]}``.

    Three other shapes arrived from live authoring and were dropped without a
    word: a bare list of ids and a bare string, which became ``{}`` at load,
    and a dict keyed ``highlight_ids``, which nothing reads. Ten of the first
    forty-nine specs carried the list form. A slide whose spec named its two
    winning strategies drew all three the same, and the critic then argued
    about highlighting the spec did carry.
    """

    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        value = {"highlight": value}
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, ids in value.items():
        name = {"highlight_ids": "highlight", "dim_ids": "dim"}.get(str(key), str(key))
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(ids, list):
            merged = list(out.get(name) or [])
            merged.extend(str(item) for item in ids if str(item).strip())
            out[name] = merged
    return out


def load_spec(payload: dict[str, Any], *, strict: bool = True,
              mode: str = "authoring") -> FigureSpec:
    """Build a ``FigureSpec`` from a JSON payload, validating it."""

    if not isinstance(payload, dict):
        raise FigureSpecError("figure spec must be a JSON object")
    if not isinstance(payload.get("synthesis_label", ""), str) or len(payload.get("synthesis_label", "")) > 64:
        raise FigureSpecError("synthesis_label must be a string of at most 64 characters", field="synthesis_label")
    grounding_payload = payload.get("grounding") or {}
    if not isinstance(grounding_payload, dict):
        raise FigureSpecError("grounding must be an object", field="grounding")
    grounding = Grounding(
        kind=str(grounding_payload.get("kind") or "schematic"),
        ledger_rows=tuple(_strings(grounding_payload.get("ledger_rows"))),
        license=str(grounding_payload.get("license") or ""),
        note=str(grounding_payload.get("note") or ""),
    )
    spec = FigureSpec(
        figure_id=str(payload.get("figure_id") or ""),
        slide_id=str(payload.get("slide_id") or ""),
        archetype=str(payload.get("archetype") or ""),
        claim=str(payload.get("claim") or ""),
        takeaway=str(payload.get("takeaway") or ""),
        title=str(payload.get("title") or ""),
        synthesis_label=payload.get("synthesis_label", "").strip(),
        grounding=grounding,
        content=payload.get("content") if isinstance(payload.get("content"), dict) else {},
        emphasis=normalise_emphasis(payload.get("emphasis")),
        section=str(payload.get("section") or ""),
        reading_direction=str(payload.get("reading_direction") or "lr"),
        size_class=str(payload.get("size_class") or "full"),
        constraints=payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {},
        required_entities=tuple(_strings(payload.get("required_entities"))),
        spec_version=str(payload.get("spec_version") or SPEC_VERSION),
    )
    if strict:
        errors = validate_spec(spec, mode=mode)
        if errors:
            raise FigureSpecError("; ".join(errors[:6]), field="content")
    return spec


def archetype_catalog() -> list[dict[str, Any]]:
    """Machine-readable archetype catalog, for spec-author prompts and docs."""

    return [
        {
            "archetype": arch.name,
            "family": arch.family,
            "summary": arch.summary,
            "required_content": list(arch.required_content),
            "grounding_kinds": list(arch.grounding_kinds),
            "demotes_to": arch.demotes_to,
            "budgets": {
                "max_primary_items": arch.max_primary,
                "max_secondary_items": arch.max_secondary,
                # The allowance at the item budget, and the curve behind it.
                # Measured, not guessed: every pair here compiles.
                "max_label_chars": arch.max_label_chars,
                "label_chars_by_item_count": [list(step) for step in arch.label_budget],
            },
        }
        for arch in sorted(ARCHETYPES.values(), key=lambda a: (a.family, a.name))
    ]
