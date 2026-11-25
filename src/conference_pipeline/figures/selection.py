"""Archetype selection, demotion and per-deck variety.

Three decisions live here, all deterministic:

1. **Selection.** A claim type proposes an archetype. The blueprint's
   ``visual_role`` maps onto a claim type, so an existing deck plan can drive
   the compiler without being rewritten.
2. **Demotion.** A quantitative archetype with no ledger rows is demoted to one
   that tells the truth (a bar chart with no data becomes a stat callout; a
   measured curve becomes a schematic curve). It is never drawn with
   illustrative values.
3. **Variety.** Per-deck quotas stop a deck from being 27 box-and-arrow
   diagrams, which is what the canary produced when every slide chose freely.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from conference_pipeline.figures.spec import (
    ARCHETYPES,
    FAMILY_OF,
    FAMILY_QUANTITATIVE,
    FAMILY_VISUAL,
    QUOTA_EXEMPT_FAMILIES,
)

# Claim type -> ordered archetype preferences.
CLAIM_TYPE_PREFERENCES: dict[str, tuple[str, ...]] = {
    "mechanism": ("flow", "nested_architecture", "loop"),
    "process": ("flow", "swimlane_sequence", "loop"),
    "architecture": ("nested_architecture", "layered_stack", "flow"),
    "comparison": ("feature_matrix", "contrast_pair", "quadrant_map"),
    "magnitude": ("paired_delta", "bar_compare", "stat_callout"),
    "result": ("bar_compare", "paired_delta", "stat_callout"),
    "behaviour_shape": ("line_curve", "schematic_curve"),
    "profile": ("radar_profile", "hierarchy_tree", "ranked_list"),
    "taxonomy": ("hierarchy_tree", "ranked_list"),
    "tradeoff": ("scatter_frontier", "quadrant_map", "tug_of_war"),
    "positioning": ("quadrant_map", "contrast_pair"),
    "sequence": ("timeline_data", "flow"),
    "temporal_video": ("video_frame_strip", "flow"),
    "image_operation": ("image_transform_strip", "flow"),
    "spatial_3d": ("spatial_3d_schematic", "nested_architecture"),
    "multimodal": ("modality_fusion", "nested_architecture"),
    "tokens": ("token_grid_overlay", "bar_compare"),
    "decision": ("decision_tree", "contrast_pair"),
    "landscape": ("landscape_regions", "quadrant_map"),
    "structure": ("hub_spoke", "hierarchy_tree"),
}

# The blueprint's existing vocabulary, so a planned deck needs no rewrite.
VISUAL_ROLE_TO_CLAIM_TYPE: dict[str, str] = {
    "mechanism_pipeline": "mechanism",
    "architecture": "architecture",
    "comparison_matrix": "comparison",
    "tradeoff_map": "tradeoff",
    "evidence_chart": "result",
    "timeline": "sequence",
    "taxonomy": "taxonomy",
    "thesis_map": "structure",
    "concept_map": "structure",
}

DEFAULT_CLAIM_TYPE = "mechanism"

# Quotas, reported in the deck manifest. Starting points, tuned after the first
# compiled deck rather than guessed at forever.
MAX_FAMILY_SHARE = 0.45
MAX_CONSECUTIVE_SAME_ARCHETYPE = 2
MIN_VISUAL_FAMILY_FIGURES_ON_VISION_DECK = 2
# A section whose slides have measured claims available should show at least
# one of them. Sections with nothing measured to show are not held to it.
MIN_QUANTITATIVE_FIGURES_PER_GROUNDED_SECTION = 1


@dataclass
class SelectionOutcome:
    """What was chosen for one slide, and why."""

    slide_id: str
    claim_type: str
    archetype: str
    proposed: str
    reason: str = ""
    demoted: bool = False
    rotated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "claim_type": self.claim_type,
            "archetype": self.archetype,
            "proposed_archetype": self.proposed,
            "demoted": self.demoted,
            "rotated_for_variety": self.rotated,
            "reason": self.reason,
        }


# What the blueprint's gloss says the figure is ABOUT, in the order it is tested.
#
# A blueprint writes `visual_role` as a coarse prefix plus a sentence of intent:
# "mechanism_pipeline: Contrast per-frame enhancement (flicker) against joint
# temporal modelling". Reading only the prefix throws away the half that knows
# this is a video figure, and every such slide collapses onto the same
# box-and-arrow diagram -- which is exactly how the canary shipped 27 of them.
#
# Each entry is (archetype, required cue words). A rule fires when the gloss
# contains any of its cues; the first firing rule wins, so the list runs from
# most specific subject to least. Deterministic, free, and easy to argue with:
# a wrong pick is a wrong word in this table, not a model's mood.
# Which claim types each archetype can honestly carry. A WEAK subject cue only
# fires when the archetype accepts the slide's claim type; a STRONG cue fires
# regardless. The distinction exists because "frames" appears in the gloss of
# almost any video slide, and matching it on a mechanism claim turned a
# seven-stage pipeline into a filmstrip -- which then drew its stage-to-stage
# edges as dashed buses over the frames, with no visible source. That was the
# single most-reported defect in the deck review.
ARCHETYPE_ACCEPTS: dict[str, frozenset[str]] = {
    "flow": frozenset({"mechanism", "process", "sequence", "image_operation", "temporal_video"}),
    "loop": frozenset({"mechanism", "process"}),
    "nested_architecture": frozenset({"mechanism", "architecture", "spatial_3d", "multimodal"}),
    "layered_stack": frozenset({"architecture", "mechanism", "structure"}),
    "branching_paths": frozenset({"mechanism", "decision", "comparison", "tradeoff", "positioning"}),
    "hierarchy_tree": frozenset({"taxonomy", "profile", "structure"}),
    "swimlane_sequence": frozenset({"process", "mechanism"}),
    "hub_spoke": frozenset({"structure", "architecture"}),
    "feature_matrix": frozenset({"comparison"}),
    "contrast_pair": frozenset({"comparison", "positioning", "tradeoff", "decision"}),
    "quadrant_map": frozenset({"tradeoff", "positioning", "comparison", "landscape"}),
    "ranked_list": frozenset({"profile", "taxonomy", "comparison"}),
    "decision_tree": frozenset({"decision", "comparison"}),
    "venn_overlap": frozenset({"comparison", "structure"}),
    "bar_compare": frozenset({"result", "magnitude", "comparison", "tokens"}),
    "paired_delta": frozenset({"magnitude", "result"}),
    "stat_callout": frozenset({"magnitude", "result", "tradeoff"}),
    "line_curve": frozenset({"behaviour_shape", "result"}),
    "scatter_frontier": frozenset({"tradeoff", "positioning", "result"}),
    "radar_profile": frozenset({"profile"}),
    "distribution_breakdown": frozenset({"result", "magnitude", "profile"}),
    "heatmap_grid": frozenset({"comparison", "result"}),
    "timeline_data": frozenset({"sequence"}),
    "magnitude_ladder": frozenset({"magnitude", "result"}),
    "schematic_curve": frozenset({"behaviour_shape"}),
    "trajectory_on_manifold": frozenset({"mechanism", "behaviour_shape"}),
    "landscape_regions": frozenset({"landscape", "tradeoff", "positioning"}),
    "tug_of_war": frozenset({"tradeoff", "mechanism"}),
    "image_transform_strip": frozenset({"image_operation", "mechanism", "process", "comparison"}),
    "video_frame_strip": frozenset({"temporal_video", "sequence"}),
    "spatial_3d_schematic": frozenset({"spatial_3d", "mechanism", "architecture"}),
    "token_grid_overlay": frozenset({"tokens", "mechanism"}),
    "modality_fusion": frozenset({"multimodal", "mechanism", "architecture"}),
    "evidence_crop": frozenset({"result"}),
    "composite": frozenset(set(CLAIM_TYPE_PREFERENCES)),
    # The slide's own text accepts no claim type: selection never routes a
    # slide to it. Only the operator's replace --text-layout outcome does.
    "key_points": frozenset(),
}

# Each entry is (archetype, strong cues, weak cues). A strong cue names the
# subject unmistakably ("flicker", "gaussian splat") and wins outright. A weak
# cue ("frames", "camera", "branch") is suggestive but common, so it only wins
# when the archetype can carry the claim the slide is making.
SUBJECT_CUES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # Family E -- the corpus is about images, video and 3-D; a box labelled
    # "image" is a poor stand-in for showing one.
    ("video_frame_strip",
     ("per-frame", "flicker", "temporal coheren", "video sequence", "motion transfer",
      "long video", "frame by frame"),
     ("frames", "across time")),
    ("spatial_3d_schematic",
     ("gaussian splat", "splat", "point cloud", "scene geometry", "mesh", "voxel",
      "3d scene", "3-d scene"),
     ("3d", "3-d", "camera", "depth map")),
    ("token_grid_overlay",
     ("visual token", "pruning tokens", "token grid", "tokens kept", "attention map"),
     ("patch", "region crop")),
    ("image_transform_strip",
     ("degrading an image", "super-resolution", "input image", "edited image",
      "image through"),
     ("blur", "crop", "watermark", "low-res", "perturb")),
    # Deliberately narrow: bare "fusion" and "multimodal" appear on most slides
    # of a multimodal deck, and matching them made every third slide the same
    # figure. Only phrases that mean modalities are actually being COMBINED.
    ("modality_fusion",
     ("modality fusion", "multimodal fusion", "fusing modalit", "fuses modalit",
      "missing modalit", "incomplete modalit", "higher-order fusion", "sensor fusion",
      "modalities converg"),
     ("event stream", "fmri")),
    # Family D -- behaviour shapes that are not charts.
    ("tug_of_war",
     ("tug-of-war", "tug of war", "pulls against", "priors versus"),
     ("competing", "balance between")),
    ("trajectory_on_manifold",
     ("manifold", "geodesic", "denoising path", "latent path"),
     ("trajector", "tangent")),
    ("landscape_regions",
     ("where it applies", "region map"),
     ("regime", "landscape", "conditional behaviour")),
    # Family A -- shapes the prefix cannot distinguish.
    ("loop",
     ("self-improv", "self-evolv", "round trip", "draft-then-verify", "loops back"),
     ("loop", "iterative", "cycle", "propos", "feedback")),
    ("swimlane_sequence",
     ("swimlane", "tool call", "back and forth"),
     ("actor", "agent and tool", "exchange")),
    ("branching_paths",
     ("two paths", "two sides", "fork", "on one side", "two parallel", "in parallel",
      "two alternative", "branching into", "branches into"),
     ("branch", "either", "one path")),
    # Strong cues only. "versus" and "against" appear in half the glosses on a
    # deck about trade-offs, and as weak cues they pulled ten slides onto one
    # archetype -- the monotony the variety quotas exist to prevent.
    ("contrast_pair",
     ("contrast between", "contrasts how", "side by side with"),
     ()),
    ("layered_stack",
     ("across layers", "layer by layer", "per layer", "stack of layers"),
     ("depth of the", "layered defense", "layered defence")),
    ("hierarchy_tree",
     ("taxonomy", "family tree", "decomposit"),
     ("categor", "abilities")),
    ("decision_tree",
     ("which method when", "when to use"),
     ("routing", "decision", "gate")),
    ("timeline_data",
     ("lineage", "timeline", "chronolog"),
     ("evolution", "generations of")),
    # Family C -- a named chart shape.
    ("radar_profile", ("radar", "multi-axis"), ("per-ability", "profile across")),
    ("distribution_breakdown", ("share of", "breakdown of"), ("proportion", "failure types")),
    ("magnitude_ladder", ("orders of magnitude",), ("parameter count", "7b vs", "model size")),
)


def parse_visual_role(visual_role: str) -> dict[str, Any]:
    """Split a blueprint ``visual_role`` into the parts an author can use.

    Blueprints write it as "prefix: gloss | components: A; B; C | edges: A -> B"
    and the whole string was being handed to the model as one line. The
    components and the edges ARE the figure's content; leaving them buried in a
    900-character string is how a spec ends up with stages the blueprint never
    named and edges it did.
    """

    text = str(visual_role or "")
    parts = [part.strip() for part in text.split("|")]
    head = parts[0] if parts else ""
    prefix, _, gloss = head.partition(":")
    out: dict[str, Any] = {
        "prefix": prefix.strip(),
        "gloss": gloss.strip() or head.strip(),
        "components": [],
        "edges": [],
        "data": [],
    }
    for part in parts[1:]:
        key, _, value = part.partition(":")
        key = key.strip().lower()
        items = [item.strip() for item in value.split(";") if item.strip()]
        if key == "components":
            out["components"] = items
        elif key == "edges":
            out["edges"] = [
                tuple(side.strip() for side in item.split("->", 1))
                for item in items if "->" in item
            ]
        elif key == "data":
            out["data"] = items
    return out


def subject_archetype(visual_role: str, *, brief: str = "", claim_type: str = "") -> str:
    """The archetype the blueprint's own words point at, or "".

    Reads the gloss after the colon (plus the figure brief, when one exists),
    not the coarse prefix. Returns "" when nothing matches, leaving the claim
    type to decide as before.

    A STRONG cue wins outright. A WEAK cue wins only if the archetype can carry
    the claim type the slide is making, and a rule that matched only on a
    rejected weak cue does not stop the scan -- the next rule still gets a turn.
    That is the fix for a seven-stage pipeline becoming a filmstrip because the
    word "frames" appeared in its gloss.
    """

    text = f"{visual_role} {brief}".lower()
    gloss = text.split(":", 1)[1] if ":" in text else text
    # Two passes, not one. A strong cue anywhere in the table beats a weak cue
    # anywhere: a gloss reading "maps the misuse landscape with defense on one
    # side and attacks on the other" matched `landscape_regions` on the weak
    # word "landscape" before `branching_paths` was ever asked about "on one
    # side", purely because it sits earlier in the list.
    for archetype, strong, _weak in SUBJECT_CUES:
        if any(cue in gloss for cue in strong):
            return archetype
    for archetype, _strong, weak in SUBJECT_CUES:
        if any(cue in gloss for cue in weak):
            if not claim_type or claim_type in ARCHETYPE_ACCEPTS.get(archetype, frozenset()):
                return archetype
    return ""


def claim_type_for(visual_role: str, *, fallback: str = DEFAULT_CLAIM_TYPE) -> str:
    """Map a blueprint ``visual_role`` onto a claim type.

    Blueprints write the role either bare ("tradeoff_map") or with a gloss
    after a colon ("tradeoff_map: thesis map connecting the clusters"), so the
    leading token is what identifies it.
    """

    role = str(visual_role or "").strip().lower()
    head = role.split(":", 1)[0].strip()
    for candidate in (role, head):
        if candidate in CLAIM_TYPE_PREFERENCES:
            return candidate
        if candidate in VISUAL_ROLE_TO_CLAIM_TYPE:
            return VISUAL_ROLE_TO_CLAIM_TYPE[candidate]
    return fallback


# How many measured pairs a shape needs before it is worth drawing. Below this
# a scatter is two dots and a line between them, which the structure lint
# refuses and the vision judge called "illustrative rather than
# publication-ready" -- so selection should not ask for one in the first place.
MIN_ROWS_FOR: dict[str, int] = {"scatter_frontier": 3, "line_curve": 3, "radar_profile": 3}


# Archetypes that place each item at an (x, y) position. Their family is not
# quantitative, so the ledger demotion below never caught them -- yet with no
# measured row the author has to invent the coordinates, and the adversarial
# critic flagged exactly that on ICML 2026 decks 01 and 02 (2026-10-01). The
# replacement says "where" as labelled regions, without fake positions.
POSITIONED_WITHOUT_MEASUREMENT: dict[str, str] = {"quadrant_map": "landscape_regions"}


def demote(archetype: str, *, has_ledger_rows: bool, ledger_rows: int | None = None) -> tuple[str, str]:
    """The archetype to draw, and why, given the measured rows available."""

    definition = ARCHETYPES.get(archetype)
    if definition is None:
        return archetype, ""
    if archetype in POSITIONED_WITHOUT_MEASUREMENT and not has_ledger_rows:
        target = POSITIONED_WITHOUT_MEASUREMENT[archetype]
        return target, (
            f"no numeric-claims ledger row supports this slide, so '{archetype}' (which places "
            f"items at exact coordinates) was demoted to '{target}' rather than drawn at "
            "invented positions"
        )
    needed = MIN_ROWS_FOR.get(archetype)
    if needed and ledger_rows is not None and ledger_rows < needed and definition.demotes_to:
        target = definition.demotes_to
        # scatter_frontier demotes to quadrant_map, which would then plot the
        # same unmeasured points at invented coordinates: with no measured row,
        # follow on to regions (with some rows the positions are real).
        if not has_ledger_rows:
            target = POSITIONED_WITHOUT_MEASUREMENT.get(target, target)
        return target, (
            f"only {ledger_rows} measured row(s) support this slide and '{archetype}' needs "
            f"{needed} to say anything; demoted to '{target}'"
        )
    if definition.family != FAMILY_QUANTITATIVE or has_ledger_rows:
        return archetype, ""
    # Follow the chain until it leaves the quantitative family. bar_compare
    # demoted to stat_callout, which needs values too, so a "result" slide
    # with no ledger row was handed an archetype it could not fill and the
    # author's spec was refused twice for missing numbers (deck 04, sl_13).
    target = definition.demotes_to
    seen = {archetype}
    while target and target not in seen and FAMILY_OF.get(target) == FAMILY_QUANTITATIVE:
        seen.add(target)
        target = ARCHETYPES[target].demotes_to
    if not target:
        return archetype, ""
    return target, (
        f"no numeric-claims ledger row supports this slide, so '{archetype}' was demoted to "
        f"'{target}' rather than drawn with illustrative values"
    )


# Picture archetypes: they show a subject and carry relations poorly -- edge
# labels are the first thing their ladder drops, and a scene routes around its
# own primitives. A brief that names this many relations, or more parts than
# a scene can place, is about the relations, and the structural archetype the
# claim type prefers says them better. Deck 03 put 8 of 21 figures on
# `spatial_3d_schematic` because every gloss on a geometry deck says "point
# cloud"; 3 of its 4 compile refusals and 5 of 9 first-round judge failures
# were those scenes, and every one re-authored cleanly as a flow or a branch.
# Per picture: the most relations it can say, and the most parts it can place.
# A strip's panels form a chain, so its natural edges are one fewer than its
# panels and only a brief with MORE relations than that -- a graph, not a
# sequence -- is about the relations. A scene places primitives in depth and
# routes arrows around them; the four scenes that failed on deck 03 carried
# 6-8 edges over 7 primitives, and the worked example carries one over three.
PICTURE_CAPACITY: dict[str, tuple[int | None, int]] = {
    "spatial_3d_schematic": (4, 5),
    "token_grid_overlay": (2, 4),
    "image_transform_strip": (None, ARCHETYPES["image_transform_strip"].max_primary),
    "video_frame_strip": (None, ARCHETYPES["video_frame_strip"].max_primary),
}
PICTURE_ARCHETYPES = frozenset(PICTURE_CAPACITY)


def relation_heavy(visual_role: str, archetype: str) -> str:
    """Why a brief is about its relations rather than its subject, or ""."""

    capacity = PICTURE_CAPACITY.get(archetype)
    if capacity is None:
        return ""
    max_edges, max_components = capacity
    role = parse_visual_role(visual_role)
    edges, components = len(role["edges"]), len(role["components"])
    if components > max_components:
        return f"the brief names {components} parts, more than a '{archetype}' can place"
    if max_edges is not None and edges > max_edges:
        return f"the brief names {edges} relations, more than a '{archetype}' can say"
    if max_edges is None and components and edges >= components:
        return f"the brief names {edges} relations over {components} parts, a graph rather than a sequence"
    return ""


MAX_FAMILY_SHARE = 0.45
FAMILY_SHARE_MIN_HISTORY = 8


# Archetypes that say something only when the brief names their subject: a
# fusion figure combines modalities, a tug of war pits two forces. Rotation
# must never fall back on one -- four pipeline slides became "modality fusion"
# the first time it could -- so they are reached by their own cue or not at all.
SUBJECT_ONLY = frozenset({
    "modality_fusion", "tug_of_war", "trajectory_on_manifold", "landscape_regions",
    "video_frame_strip", "token_grid_overlay",
})


def labels_too_long(visual_role: str, archetype: str) -> str:
    """Why this archetype cannot hold the brief's own component names, or "".

    The brief's components become the figure's labels, and every archetype
    publishes how long a label may be at a given item count. Choosing one whose
    budget those names already exceed asks the author for something the
    validator will reject -- and it did, three times each on three decks, where
    a five-region landscape allows 20 characters and the briefs named regions
    like "Memory budget (high to low)".
    """

    definition = ARCHETYPES.get(archetype)
    if definition is None:
        return ""
    components = [str(item).strip() for item in parse_visual_role(visual_role)["components"] if str(item).strip()]
    if not components:
        return ""
    allowance = definition.label_chars(len(components))
    longest = max(components, key=len)
    if len(longest) <= allowance:
        return ""
    return (f"the brief names {len(components)} components and its longest, {longest[:34]!r}, is "
            f"{len(longest)} characters against the {allowance} '{archetype}' allows at that count")


def _alternatives(visual_role: str) -> list[str]:
    """Where rotation looks for another archetype, in order: the structural
    shapes named in the subject table, then the pictures -- and a picture only
    when the brief is within what it can place and say. Walking the subject
    table in its own order handed every rotated pipeline to the 3-D scene,
    because the scene sits near the top and accepts 'mechanism'."""

    cues = [row[0] for row in SUBJECT_CUES if row[0] not in SUBJECT_ONLY]
    structural = [name for name in cues if name not in PICTURE_ARCHETYPES]
    pictures = [name for name in cues
                if name in PICTURE_ARCHETYPES and not relation_heavy(visual_role, name)]
    return [*structural, *pictures]


def _crowded_family(archetype: str, history: list[str]) -> str:
    """The family this pick would keep over the deck quota, or ""."""

    history = quota_counted(history)
    if len(history) < FAMILY_SHARE_MIN_HISTORY:
        return ""
    family = FAMILY_OF.get(archetype, "")
    if not family or family in QUOTA_EXEMPT_FAMILIES:
        return ""
    same = sum(1 for item in history if FAMILY_OF.get(item) == family)
    return family if (same + 1) / (len(history) + 1) > MAX_FAMILY_SHARE else ""


def select_archetype(
    *,
    slide_id: str,
    visual_role: str = "",
    claim_type: str = "",
    has_ledger_rows: bool = False,
    ledger_rows: int | None = None,
    history: list[str] | None = None,
    preferred: str = "",
    brief: str = "",
) -> SelectionOutcome:
    """Choose one slide's archetype, applying demotion and archetype rotation.

    Order of authority: an explicit ``preferred`` archetype, then what the
    blueprint's own gloss says the figure is about, then the claim type's first
    preference. The middle step is what stops a deck of twenty-five slides from
    collapsing onto three archetypes: the coarse role prefix is the same for a
    video figure, a 3-D figure and a block diagram, and only the gloss knows
    which is which.
    """

    resolved_claim_type = claim_type or claim_type_for(visual_role)
    options = CLAIM_TYPE_PREFERENCES.get(resolved_claim_type, CLAIM_TYPE_PREFERENCES[DEFAULT_CLAIM_TYPE])
    subject = subject_archetype(visual_role, brief=brief, claim_type=resolved_claim_type)
    subject_reason = ""
    heavy = relation_heavy(visual_role, subject) if subject in PICTURE_ARCHETYPES else ""
    if not heavy and subject:
        heavy = labels_too_long(visual_role, subject)
    if preferred in ARCHETYPES:
        proposed = preferred
    elif subject and heavy:
        proposed = options[0]
        subject_reason = (
            f"the brief describes a '{subject}' subject, but {heavy}; a picture shows a "
            f"subject and says relations poorly, so the '{resolved_claim_type}' claim type's "
            f"'{proposed}' carries them"
        )
    elif subject:
        proposed = subject
        subject_reason = (
            f"the slide's own figure brief describes a '{subject}' subject, which the "
            f"'{resolved_claim_type}' claim type alone would not have chosen"
        )
    else:
        proposed = options[0]

    # ...and whichever archetype we land on must be able to hold the names.
    if not preferred:
        for candidate in (proposed, *options, *_alternatives(visual_role)):
            if not labels_too_long(visual_role, candidate):
                if candidate != proposed:
                    subject_reason = (
                        f"'{proposed}' cannot hold the brief's labels ("
                        f"{labels_too_long(visual_role, proposed)}), so this slide uses '{candidate}'"
                    )
                    proposed = candidate
                break

    archetype, reason = demote(proposed, has_ledger_rows=has_ledger_rows, ledger_rows=ledger_rows)
    demoted = archetype != proposed
    if not reason and subject_reason:
        reason = subject_reason

    rotated = False
    recent = (history or [])[-MAX_CONSECUTIVE_SAME_ARCHETYPE:]
    if len(recent) == MAX_CONSECUTIVE_SAME_ARCHETYPE and all(item == archetype for item in recent):
        for candidate in (*options, *_alternatives(visual_role)):
            alternative, _ = demote(candidate, has_ledger_rows=has_ledger_rows,
                                    ledger_rows=ledger_rows)
            if alternative != archetype:
                reason = (
                    f"'{archetype}' had already run for {MAX_CONSECUTIVE_SAME_ARCHETYPE} consecutive "
                    f"slides, so this slide uses '{alternative}'"
                )
                archetype = alternative
                rotated = True
                break
    # The deck quota caps any one family at MAX_FAMILY_SHARE of the figures,
    # and it used to be checked only after every spec had been paid for. Once
    # enough of the deck has been chosen to know the share, a pick that would
    # keep an over-represented family over the cap yields to the first
    # preference from another family that can carry the claim type.
    crowded = _crowded_family(archetype, history or [])
    if crowded:
        for candidate in (*options, *_alternatives(visual_role)):
            alternative, _ = demote(candidate, has_ledger_rows=has_ledger_rows,
                                    ledger_rows=ledger_rows)
            if (alternative != archetype
                    and FAMILY_OF.get(alternative) != crowded
                    and resolved_claim_type in ARCHETYPE_ACCEPTS.get(alternative, frozenset())
                    and not _crowded_family(alternative, history or [])):
                reason = (
                    f"'{archetype}' would keep the '{crowded}' family over "
                    f"{int(MAX_FAMILY_SHARE * 100)}% of the deck, so this slide uses '{alternative}'"
                )
                archetype = alternative
                rotated = True
                break
        else:
            # Nothing outside the family can carry this claim. Keep the
            # archetype and say so; the deck's variety report will name the
            # share, and whether to accept it is the operator's call.
            reason = (reason + "; " if reason else "") + (
                f"the '{crowded}' family is over {int(MAX_FAMILY_SHARE * 100)}% of the deck and "
                f"no other family carries a '{resolved_claim_type}' claim"
            )

    return SelectionOutcome(
        slide_id=slide_id,
        claim_type=resolved_claim_type,
        archetype=archetype,
        proposed=proposed,
        reason=reason,
        demoted=demoted,
        rotated=rotated,
    )


def quota_counted(archetypes: list[str]) -> list[str]:
    """The archetypes the variety quotas count: everything but the slide-text
    family, which is the operator's replacement for a figure, not a figure."""

    return [name for name in archetypes if FAMILY_OF.get(name) not in QUOTA_EXEMPT_FAMILIES]


@dataclass
class DeckVarietyReport:
    """Whether a deck's figure mix meets the variety quotas."""

    total: int
    family_counts: dict[str, int] = field(default_factory=dict)
    archetype_counts: dict[str, int] = field(default_factory=dict)
    section_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    # Slides carried by an exempt family (the slide's own text), by archetype.
    exempt_counts: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_figures": self.total,
            "family_counts": self.family_counts,
            "archetype_counts": self.archetype_counts,
            "section_counts": self.section_counts,
            "exempt_from_quotas": self.exempt_counts,
            "quotas": {
                "max_family_share": MAX_FAMILY_SHARE,
                "max_consecutive_same_archetype": MAX_CONSECUTIVE_SAME_ARCHETYPE,
                "min_visual_family_figures_on_vision_deck": MIN_VISUAL_FAMILY_FIGURES_ON_VISION_DECK,
                "min_quantitative_figures_per_grounded_section": MIN_QUANTITATIVE_FIGURES_PER_GROUNDED_SECTION,
            },
            "findings": self.findings,
            "passed": self.passed,
        }


def deck_variety_report(
    archetypes: list[str],
    *,
    vision_deck: bool = False,
    sections: list[str] | None = None,
    grounded_sections: set[str] | None = None,
) -> DeckVarietyReport:
    """Check a whole deck's figure mix against the quotas.

    Slides in ``QUOTA_EXEMPT_FAMILIES`` are reported under ``exempt_counts``
    and removed before any quota is computed, together with their section
    entries, so a text-layout slide neither dilutes a family's share nor
    breaks or extends a run of the same archetype.
    """

    exempt_counts = dict(sorted(Counter(
        name for name in archetypes if FAMILY_OF.get(name) in QUOTA_EXEMPT_FAMILIES
    ).items()))
    if sections:
        sections = [section for name, section in zip(archetypes, sections)
                    if FAMILY_OF.get(name) not in QUOTA_EXEMPT_FAMILIES]
    archetypes = quota_counted(archetypes)
    total = len(archetypes)
    families = [FAMILY_OF.get(name, "unknown") for name in archetypes]
    family_counts = dict(sorted(Counter(families).items()))
    archetype_counts = dict(sorted(Counter(archetypes).items()))
    findings: list[str] = []

    if total:
        for family, count in family_counts.items():
            share = count / total
            if share > MAX_FAMILY_SHARE:
                findings.append(
                    f"family '{family}' covers {count}/{total} figures ({share:.0%}), above the "
                    f"{MAX_FAMILY_SHARE:.0%} quota; vary the claim types or the archetypes chosen for them"
                )
    run = 1
    for index in range(1, total):
        if archetypes[index] == archetypes[index - 1]:
            run += 1
            if run > MAX_CONSECUTIVE_SAME_ARCHETYPE:
                findings.append(
                    f"'{archetypes[index]}' runs for {run} consecutive slides ending at slide {index + 1}; "
                    f"the limit is {MAX_CONSECUTIVE_SAME_ARCHETYPE}"
                )
                break
        else:
            run = 1
    # Per-section quota: a section whose slides have measured claims available
    # should show at least one of them, rather than narrating every number while
    # every figure stays a box-and-arrow diagram.
    section_counts: dict[str, dict[str, int]] = {}
    if sections:
        for index, section in enumerate(sections):
            if index >= total:
                break
            key = str(section or "").strip() or "(unsectioned)"
            bucket = section_counts.setdefault(key, {})
            family = FAMILY_OF.get(archetypes[index], "unknown")
            bucket[family] = bucket.get(family, 0) + 1
        section_counts = {key: dict(sorted(value.items())) for key, value in sorted(section_counts.items())}
        for key in sorted(grounded_sections or set()):
            bucket = section_counts.get(key)
            if bucket is None:
                continue
            quantitative = bucket.get(FAMILY_QUANTITATIVE, 0)
            if quantitative < MIN_QUANTITATIVE_FIGURES_PER_GROUNDED_SECTION:
                findings.append(
                    f"section '{key}' has measured claims available but carries {quantitative} "
                    f"quantitative figures; at least {MIN_QUANTITATIVE_FIGURES_PER_GROUNDED_SECTION} "
                    "is expected so the section shows a number rather than only saying one"
                )

    if vision_deck:
        visual = family_counts.get(FAMILY_VISUAL, 0)
        if visual < MIN_VISUAL_FAMILY_FIGURES_ON_VISION_DECK:
            findings.append(
                f"a vision deck carries {visual} visual-domain figures; at least "
                f"{MIN_VISUAL_FAMILY_FIGURES_ON_VISION_DECK} are expected (frames, images, 3-D, tokens)"
            )
    return DeckVarietyReport(
        total=total, family_counts=family_counts, archetype_counts=archetype_counts,
        section_counts=section_counts, findings=findings, exempt_counts=exempt_counts,
    )
