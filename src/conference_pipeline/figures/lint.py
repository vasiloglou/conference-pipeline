"""Does this spec's STRUCTURE say what its claim says?

The geometry gate answers a different question: is the picture legal. A figure
can be flawless by that measure and still state the wrong thing -- alternatives
drawn as a sequence, a pipeline drawn as a filmstrip, two incommensurable
numbers on one axis, a node connected to nothing. Every one of those shipped on
the canary decks and every one drew a vision-judge finding that no geometry
check could ever have caught, because nothing on the canvas was broken.

These rules are deterministic and read the spec, not the pixels. They run
before the compiler, which means a defect costs one seven-cent re-author rather
than a paid round of rendering and judging.

Three outcomes:

* ``error``  -- refuse. The spec says something untrue or unreadable.
* ``demote`` -- refuse, naming the archetype that fits the content.
* ``warn``   -- record in the manifest; a reviewer decides.

Two modes. Under ``authoring`` the soft rules are errors, because the author is
still in the loop and a fix is cheap. Under ``compile`` they are warnings: by
then the only remedy is re-authoring, and blocking a deck on a judgement call
is not the compiler's decision to make.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from conference_pipeline.figures.spec import FigureSpec

LINT_VERSION = "figure-lint/1"


@dataclass(frozen=True)
class LintFinding:
    code: str
    severity: str            # error | demote | warn
    message: str
    demote_to: str = ""
    suggested_change: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "severity": self.severity, "message": self.message}
        if self.demote_to:
            payload["demote_to"] = self.demote_to
        if self.suggested_change:
            payload["suggested_change"] = self.suggested_change
        return payload

    def __str__(self) -> str:
        return f"{self.severity}:{self.code}: {self.message}"


# Units that name a measured quantity. An axis in "ordinal position" or
# "relative improvement" is not measuring anything, so points on it cannot be
# compared and a frontier through them means nothing.
MEASURED_UNITS = frozenset({
    "%", "pp", "x", "×", "ms", "s", "sec", "seconds", "min", "hours", "fps",
    "gb", "mb", "kb", "b", "m", "k", "db", "px", "tokens", "steps", "epochs",
    "params", "flops", "images", "frames", "samples", "queries", "points",
})
ORDINAL_UNITS = frozenset({
    "ordinal position", "ordinal", "rank", "stage", "position", "index",
    "order", "relative improvement", "level", "tier", "category",
})

# Words that name a step in a process rather than a moment in time. A filmstrip
# whose frames are "reward signal" and "evaluation benchmark" is a pipeline
# wearing a filmstrip's clothes.
PIPELINE_NOUNS = (
    "stage", "module", "branch", "constraint", "reward", "framework",
    "benchmark", "evaluation", "pipeline", "encoder", "decoder", "backbone",
    "objective", "loss", "sampler", "adapter", "policy", "head",
)

# Phrases that mean "these are alternatives", which a single strip cannot say.
# Deliberately without a bare "while": "a 4K edit while cutting memory 56%" is
# simultaneity, not contrast, and reading it as contrast flagged a genuine
# three-step sequence. The claims that really are contrasts say so another way
# too -- "on one side", "instead", or two families of panel id.
ALTERNATIVE_MARKERS = (" or ", " vs ", " versus ", "alternative", "instead",
                       "whereas", "two sides", "opposite", "on one side",
                       "on the other hand")


def _labels(node: Any, keys: Sequence[str] = ("label", "caption")) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in keys and isinstance(value, str) and value.strip():
                found.append(value)
            else:
                found.extend(_labels(value, keys))
    elif isinstance(node, list):
        for item in node:
            found.extend(_labels(item, keys))
    return found


def _walk_children(nodes: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [node for node in nodes if isinstance(node, dict)]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(child for child in (node.get("children") or []) if isinstance(child, dict))
    return out


def _unit(axis: Any) -> str:
    if isinstance(axis, dict):
        return str(axis.get("unit") or "").strip().lower()
    return ""


# -- rules ---------------------------------------------------------------------

def _minimum_counts(spec: FigureSpec) -> list[LintFinding]:
    content, name = spec.content, spec.archetype
    findings: list[LintFinding] = []

    # Minimums the LAYOUT enforced and the validator did not, so the author
    # only learned after the authoring call was spent. A two-set venn and a
    # two-actor swimlane are the shapes those layouts can draw; below that the
    # spec is a different figure, and saying so here costs nothing.
    if name == "venn_overlap":
        sets = [item for item in (content.get("sets") or []) if isinstance(item, dict)]
        if len(sets) < 2:
            findings.append(LintFinding(
                "venn_needs_two_sets", "demote",
                f"a venn of {len(sets)} set says nothing about an overlap",
                demote_to="stat_callout",
                suggested_change="name the second set, or state the one fact as a callout",
            ))
    if name == "swimlane_sequence":
        actors = [item for item in (content.get("actors") or []) if isinstance(item, dict)]
        if len(actors) < 2:
            findings.append(LintFinding(
                "swimlane_needs_two_actors", "demote",
                f"a swimlane with {len(actors)} actor has no exchange to show",
                demote_to="flow",
                suggested_change="name the second actor, or use flow for a single actor's steps",
            ))

    if name == "loop":
        nodes = [node for node in (content.get("nodes") or []) if isinstance(node, dict)]
        if len(nodes) < 3:
            findings.append(LintFinding(
                "loop_too_short", "demote",
                f"a cycle of {len(nodes)} reads as a pair, not a loop",
                demote_to="flow" if len(nodes) > 1 else "stat_callout",
                suggested_change="use flow for a two-step process, or name the third stage",
            ))

    if name == "loop":
        content_output = content.get("output") or {}
        if not str(content_output.get("label") or content.get("centre_label") or "").strip():
            findings.append(LintFinding(
                "loop_without_an_output", "warn",
                "the cycle names nothing that leaves it, so a reader sees a machine "
                "running and never learns what it produces",
                suggested_change='add content.output {label, exit_from} naming what the loop yields',
            ))
        node_ids = {str(node.get("id") or "") for node in (content.get("nodes") or [])
                    if isinstance(node, dict)}
        exit_from = str(content_output.get("exit_from") or "").strip()
        if exit_from and exit_from not in node_ids:
            findings.append(LintFinding(
                "loop_exit_unknown", "error",
                f"content.output.exit_from names {exit_from!r}, which is not a node in this loop",
            ))

    if name in {"hierarchy_tree", "decision_tree"}:
        children = [c for c in (content.get("children") or []) if isinstance(c, dict)]
        if len(children) < 2:
            findings.append(LintFinding(
                "tree_without_branches", "demote",
                f"a tree with {len(children)} branch does not decompose anything",
                demote_to="flow",
                suggested_change="a taxonomy needs at least two groups under its root",
            ))

    if name == "scatter_frontier":
        points = [p for p in (content.get("points") or []) if isinstance(p, dict)]
        if len(points) < 3:
            findings.append(LintFinding(
                "scatter_too_few_points", "demote",
                f"{len(points)} measured pair(s) do not make a scatter, and a line "
                "through two of them is not a frontier",
                demote_to="stat_callout" if len(points) < 2 else "paired_delta",
                suggested_change="show the numbers as callouts, or find more measured pairs",
            ))
        unlabelled = [p for p in points if not str(p.get("label") or "").strip()]
        if unlabelled:
            findings.append(LintFinding(
                "scatter_point_unnamed", "error",
                f"{len(unlabelled)} point(s) carry no label; a dot nobody can name "
                "tells a reader nothing",
            ))
        if content.get("frontier"):
            on_frontier = [p for p in points if p.get("on_frontier")]
            if len(on_frontier) < 3:
                findings.append(LintFinding(
                    "frontier_too_few_points", "warn",
                    f"a Pareto frontier drawn through {len(on_frontier)} point(s) is a "
                    "line between them, not a frontier",
                    suggested_change="drop content.frontier, or mark the points that define it",
                ))
    return findings


def _orphans(spec: FigureSpec) -> list[LintFinding]:
    """Nodes joined to nothing. The reader is left asking what they connect to."""

    if spec.archetype not in {"flow", "nested_architecture"}:
        return []
    nodes = [node for node in (spec.content.get("nodes") or []) if isinstance(node, dict)]
    if len(nodes) < 2:
        return []
    joined: set[str] = set()
    for edge in (spec.content.get("edges") or []):
        if isinstance(edge, dict):
            joined.add(str(edge.get("from") or ""))
            joined.add(str(edge.get("to") or ""))
    grouped: set[str] = set()
    for group in (spec.content.get("groups") or []):
        if isinstance(group, dict):
            grouped.update(str(member) for member in (group.get("members") or []))

    orphans = [
        str(node.get("label") or node.get("id"))
        for node in nodes
        # A node inside a group is contained, not orphaned: the container says
        # what it belongs to even when no arrow does.
        if str(node.get("id") or "") not in joined and str(node.get("id") or "") not in grouped
    ]
    if not orphans:
        return []
    return [LintFinding(
        "orphan_node", "error",
        f"{', '.join(repr(name) for name in orphans[:3])} joins nothing; every node in a "
        "mechanism is reached by an edge, or belongs to a group that explains it",
    )]


def _flow_relations(spec: FigureSpec) -> list[LintFinding]:
    """Flag causal arrows that leave a flow's return or merge implicit."""

    if spec.archetype != "flow":
        return []
    nodes = [node for node in (spec.content.get("nodes") or []) if isinstance(node, dict)]
    edges = [edge for edge in (spec.content.get("edges") or []) if isinstance(edge, dict)]
    causal = [edge for edge in edges
              if str(edge.get("kind") or "flow") not in {"feedback", "parallel", "contrast"}]
    incoming: dict[str, set[str]] = {}
    adjacency: dict[str, set[str]] = {}
    for edge in causal:
        source, target = str(edge.get("from") or ""), str(edge.get("to") or "")
        incoming.setdefault(target, set()).add(source)
        adjacency.setdefault(source, set()).add(target)
    findings = []
    unnamed = [str(node.get("label") or node.get("id") or "") for node in nodes
               if len(incoming.get(str(node.get("id") or ""), set())) >= 2
               and node.get("kind") != "merge"]
    if unnamed:
        findings.append(LintFinding(
            "flow_implicit_merge", "warn",
            f"{', '.join(repr(name) for name in unnamed[:3])} receives multiple causal paths "
            "without an explicit merge role",
            suggested_change="mark the joining node kind: 'merge' and name what combines",
        ))
    for edge in causal:
        source, target = str(edge.get("from") or ""), str(edge.get("to") or "")
        seen: set[str] = set()
        pending = [target]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, ()))
        if source in seen:
            findings.append(LintFinding(
                "flow_implicit_feedback", "warn",
                "causal arrows form a cycle without an explicit feedback edge",
                suggested_change="mark the return edge kind: 'feedback' so it gets an outer lane",
            ))
            break
    return findings


def _branch_labels(spec: FigureSpec) -> list[LintFinding]:
    if spec.archetype != "decision_tree":
        return []
    root = spec.content.get("root") or {}
    children = [c for c in (spec.content.get("children") or []) if isinstance(c, dict)]
    if not children:
        return []
    unlabelled = [
        c for c in children
        if not str(c.get("branch_label") or c.get("edge_label") or c.get("condition") or "").strip()
    ]
    if not unlabelled:
        return []
    return [LintFinding(
        "decision_branch_unlabelled", "warn",
        f"{len(unlabelled)} branch(es) below {str(root.get('label') or 'the question')!r} carry "
        "no branch_label, so the picture asks a question and does not say which answer "
        "leads where",
        suggested_change='give each child a branch_label: "yes"/"no", or the condition that routes there',
    )]


def _commensurable(spec: FigureSpec) -> list[LintFinding]:
    """Two axes that measure nothing comparable make a chart that means nothing."""

    if spec.archetype != "scatter_frontier":
        return []
    findings: list[LintFinding] = []
    for axis_key in ("x", "y"):
        unit = _unit(spec.content.get(axis_key))
        if not unit:
            findings.append(LintFinding(
                "axis_unit_missing", "error",
                f"the {axis_key} axis names no unit; a measured chart states what it measures",
            ))
        elif unit in ORDINAL_UNITS:
            findings.append(LintFinding(
                "axis_not_measured", "demote",
                f"the {axis_key} axis is in {unit!r}, which is an ordering, not a measurement; "
                "points on it cannot be compared and a frontier through them means nothing",
                demote_to="quadrant_map",
                suggested_change="plot a measured quantity, or use quadrant_map for a positioning claim",
            ))
    return findings


def _mixed_metrics(spec: FigureSpec) -> list[LintFinding]:
    """Point labels that each name their own metric pair, sharing one axis."""

    if spec.archetype != "scatter_frontier":
        return []
    points = [p for p in (spec.content.get("points") or []) if isinstance(p, dict)]
    self_describing = [
        str(p.get("label") or "") for p in points
        if re.search(r"\b(vs\.?|versus)\b", str(p.get("label") or ""), re.IGNORECASE)
    ]
    if len(self_describing) < 2:
        return []
    return [LintFinding(
        "points_measure_different_things", "warn",
        f"{len(self_describing)} points name their own metric pair "
        f"({self_describing[0][:48]!r}, ...), so the axes mean something different for each "
        "one and their positions cannot be compared",
        suggested_change="plot one metric pair per figure, or show the numbers as callouts",
    )]


def _frame_strip_is_a_pipeline(spec: FigureSpec) -> list[LintFinding]:
    """A filmstrip says "these are moments in one sequence". Stages are not."""

    if spec.archetype != "video_frame_strip":
        return []
    frames = [f for f in (spec.content.get("frames") or []) if isinstance(f, dict)]
    captions = [str(f.get("caption") or "").lower() for f in frames]
    time_label = str(spec.content.get("time_label") or "").lower()
    hits = [
        caption for caption in captions
        if any(noun in caption for noun in PIPELINE_NOUNS)
    ]
    findings: list[LintFinding] = []
    if len(hits) >= 2 or any(noun in time_label for noun in ("stage", "pipeline", "module")):
        findings.append(LintFinding(
            "frame_strip_is_a_pipeline", "demote",
            "the frames name stages of a process, not moments of one sequence "
            f"(time axis {time_label!r}); a filmstrip claims the frames are the same "
            "scene over time",
            demote_to="flow",
            suggested_change="use flow, whose arrows carry the relation between stages",
        ))

    for link in (spec.content.get("links") or []):
        if not isinstance(link, dict):
            continue
        try:
            start, end = int(link.get("from", 0)), int(link.get("to", 0))
        except (TypeError, ValueError):
            continue
        if abs(end - start) != 1:
            findings.append(LintFinding(
                "frame_link_skips", "error",
                f"a link joins frame {start} to frame {end}; a filmstrip's links join "
                "NEIGHBOURING frames, and a claim that jumps between stages is a flow",
                suggested_change="use flow, or link adjacent frames only",
            ))
            break
    return findings


def _alternatives_as_sequence(spec: FigureSpec, deck_claims: Sequence[str]) -> list[LintFinding]:
    """A strip is one thing happening after another. Two outcomes are not."""

    if spec.archetype != "image_transform_strip":
        return []
    panels = [p for p in (spec.content.get("panels") or []) if isinstance(p, dict)]
    if len(panels) < 3:
        return []
    haystack = " ".join([
        *(str(p.get("caption") or "") for p in panels),
        spec.claim,
        *(str(claim) for claim in deck_claims),
    ]).lower()
    prefixes: dict[str, int] = {}
    for panel in panels:
        head = str(panel.get("id") or "").split("_")[0]
        if head:
            prefixes[head] = prefixes.get(head, 0) + 1
    branched = sum(1 for count in prefixes.values() if count >= 2) >= 2
    worded = any(marker in haystack for marker in ALTERNATIVE_MARKERS)
    if not (branched or worded):
        return []
    return [LintFinding(
        "alternatives_drawn_as_a_sequence", "demote",
        "the panels are two alternatives, and a strip draws them as one thing after "
        "another -- so the picture says the second outcome FOLLOWS the first",
        demote_to="branching_paths",
        suggested_change="use branching_paths for one input with two outcomes, or contrast_pair",
    )]


def _notes_are_numbers(spec: FigureSpec) -> list[LintFinding]:
    if spec.archetype != "token_grid_overlay":
        return []
    notes = [n for n in (spec.content.get("notes") or []) if isinstance(n, dict)]
    wordy = [n for n in notes if not re.search(r"\d", str(n.get("value") or ""))]
    if not wordy or len(wordy) < len(notes):
        return []
    return [LintFinding(
        "token_notes_are_a_progression", "warn",
        "the notes beside the grid carry stage names rather than measured values, so the "
        "grid and the notes are two unrelated pictures on one slide",
        suggested_change="notes are numeric callouts about the grid; a progression belongs in a flow",
    )]


def _fusion_alternatives(spec: FigureSpec) -> list[LintFinding]:
    """Competing fusion mechanisms: two or three, each named, each with a result.

    One alternative is not a comparison, and four do not fit beside their
    outputs in the plot's height. The author is told to cut; the compiler does
    not shrink a comparison into illegibility to keep the count.
    """

    if spec.archetype != "modality_fusion":
        return []
    fusion = spec.content.get("fusion") or {}
    if not isinstance(fusion, dict):
        return []
    findings: list[LintFinding] = []
    alternatives = fusion.get("alternatives")
    if isinstance(alternatives, list):
        items = [a for a in alternatives if isinstance(a, dict)]
        if len(items) == 1:
            findings.append(LintFinding(
                "fusion_alternatives_count", "error",
                "one fusion alternative is not a comparison",
                suggested_change="name the competing mechanism too, or drop `alternatives` "
                                 "and put the mechanism in fusion.label",
            ))
        elif len(items) > 3:
            findings.append(LintFinding(
                "fusion_alternatives_count", "error",
                f"{len(items)} fusion alternatives will not fit beside their outputs; three is the most",
                suggested_change="keep the three the claim contrasts",
            ))
        for item in items:
            if not str(item.get("label") or "").strip():
                findings.append(LintFinding(
                    "fusion_alternative_unlabelled", "error",
                    "every fusion alternative needs a label naming the mechanism",
                    suggested_change="add `label` to each entry of fusion.alternatives",
                ))
                break
    outputs = fusion.get("output")
    if isinstance(outputs, list) and len(outputs) > 3:
        findings.append(LintFinding(
            "fusion_outputs_count", "error",
            f"{len(outputs)} fusion outputs will not stack in the plot; three is the most",
            suggested_change="keep the outputs the takeaway names",
        ))
    return findings


_RULES = (
    _minimum_counts,
    _orphans,
    _flow_relations,
    _branch_labels,
    _commensurable,
    _mixed_metrics,
    _frame_strip_is_a_pipeline,
    _notes_are_numbers,
    _fusion_alternatives,
)

# Codes the compile route records rather than refuses. Every one of them is a
# judgement about what a reader will conclude, and by compile time the only
# remedy is re-authoring the slide -- which is the operator's call, not the
# compiler's.
SOFT_CODES = frozenset({
    "decision_branch_unlabelled",
    "frontier_too_few_points",
    "points_measure_different_things",
    "token_notes_are_a_progression",
    "loop_without_an_output",
})


def _title_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def title_repeats_slide_title(spec: FigureSpec, slide_title: str) -> list[LintFinding]:
    """A figure heading that only repeats the slide's own title tells the reader nothing new.

    The slide already shows its title above the figure; a heading that says the same
    thing again reads as a layout glitch (11 of 26 slides on ICML 2026 deck 01, and a
    vision reviewer flagged it). The heading is where the figure names its mechanism
    or its measurement. Text layouts carry the slide's own words by design and are exempt.

    Not part of ``lint_spec``: it needs the slide, which a spec alone does not carry.
    """

    from conference_pipeline.figures.spec import ARCHETYPES, FAMILY_TEXT

    arch = ARCHETYPES.get(spec.archetype)
    if arch is not None and arch.family == FAMILY_TEXT:
        return []
    heading, slide = _title_key(spec.title), _title_key(slide_title)
    if not heading or heading != slide:
        return []
    return [LintFinding(
        "figure_title_repeats_slide_title", "warn",
        f"the figure title {spec.title!r} repeats the slide's own title",
        suggested_change="title the figure by its mechanism or measurement (what is drawn), "
                         "not by what the slide is about",
    )]


_OUTCOME_PATTERNS = tuple(re.compile(r"\b" + stem + r"\w*\b", re.I)
                          for stem in ("certif", "guarantee", "recover", "intact", "unchanged", "proves", "eliminates"))


def unsupported_outcomes(spec: FigureSpec, slide: dict[str, Any]) -> list[LintFinding]:
    """Require bound contribution notes for strong outcome predicates.

    A lexical check is a conservative trigger; the critic checks meaning,
    negation, attribution and position-paper status.
    """
    rows = slide.get("approved_sources")
    if rows is None:
        rows = slide.get("speaker_author_work_mentions") or slide.get("named_paper_talk_speaker_author_mentions") or []
    # A bound work's TITLE is evidence too: a predicate that names the work's
    # own subject ("Position: Certified Correctness in Neural Reasoning") is
    # not an outcome the figure invented. Checking notes alone blocked ICML
    # 2026 deck 03's hedged "runtime certification (argued by ...)" slides.
    notes = " ".join(
        f"{row.get('contribution_note') or ''} {row.get('title') or ''}"
        for row in rows if isinstance(row, dict))
    # A string copied verbatim from the approved slide (a text layout's points,
    # its claim's first sentence) is not an outcome the figure invented; the
    # slide text is judged by Gate A. Rejecting it made the author redraw ICML
    # 2026 deck 10/11 text layouts and reintroduce fixed errors.
    slide_text = " ".join(" ".join(str(item).split()) for item in [
        slide.get("title") or "", slide.get("display_title") or "",
        *(slide.get("visible_text") or []), *(slide.get("claims") or [])])
    texts = [text for text in (spec.claim, spec.takeaway, *_labels(spec.content))
             if not (text and " ".join(text.split()) in slide_text)]
    return [LintFinding("unsupported_outcome", "error",
                        f"outcome predicate {match.group(0)!r} has no bound source note",
                        suggested_change="remove the outcome or state only what the source proposes")
            for text in texts for pattern in _OUTCOME_PATTERNS
            if (match := pattern.search(text)) and not pattern.search(notes)]


def lint_spec(
    spec: FigureSpec,
    *,
    deck_claims: Sequence[str] = (),
    mode: str = "compile",
) -> list[LintFinding]:
    """Every structural finding for one spec, in a fixed order."""

    findings: list[LintFinding] = []
    for rule in _RULES:
        findings.extend(rule(spec))
    findings.extend(_alternatives_as_sequence(spec, deck_claims))
    if mode == "authoring":
        # The author is still in the loop and a fix costs one call, so the soft
        # rules are errors there. At compile time they are advice.
        findings = [
            LintFinding(f.code, "error" if f.severity == "warn" else f.severity,
                        f.message, f.demote_to, f.suggested_change)
            for f in findings
        ]
    return findings


def rule_summaries() -> list[str]:
    """One line per rule, for the authoring prompt.

    Rendered from the code so the prompt and the checks cannot drift apart.
    """

    return [
        "A loop needs at least three stages; two is a flow.",
        "A venn needs two or three sets; a swimlane needs two or more actors.",
        "A loop names what leaves it: content.output {label, exit_from}.",
        "A tree needs at least two branches under its root.",
        "A scatter needs at least three measured pairs, each with a label naming its method.",
        "Scatter axes state a measured unit (%, x, ms, GB). An ordering is not a measurement.",
        "Every node in a mechanism is reached by an edge, or belongs to a group.",
        "A decision's branches each carry a branch_label: the answer that routes there.",
        "The figure title names what is drawn (a mechanism or a measurement); it never repeats the slide's own title.",
        "A fusion with competing mechanisms lists two or three fusion.alternatives, each with a "
        "label and an output; fusion.output may be one result or a list of up to three.",
        "A filmstrip's frames are moments of one sequence and its links join NEIGHBOURS. "
        "Stages of a process are a flow.",
        "Alternatives are never a sequence: two outcomes from one input are branching_paths.",
        "Notes beside a token grid are measured callouts, not stage names.",
    ]


__all__ = ["LINT_VERSION", "LintFinding", "lint_spec", "rule_summaries"]


def source_fidelity_errors(spec: FigureSpec, slide: dict[str, Any]) -> list[LintFinding]:
    """Slide-aware composition, negative-cell and absence checks."""
    from conference_pipeline.figures.grounding import component_sources
    from conference_pipeline.presentation.claim_fidelity import absence_claims
    rows = [r for r in (slide.get('approved_sources') or slide.get('speaker_author_work_mentions') or []) if isinstance(r, dict)]
    findings = []
    def error(code: str, message: str) -> None:
        findings.append(LintFinding(code, 'error', message))
    for text in [spec.claim, spec.takeaway, *_labels(spec.content)]:
        if absence_claims(text):
            error('absence_claim', 'paper-wide absence assertion: ' + text)
    ids: dict[str, set[str]] = {}
    joined = False
    def walk(node: Any) -> set[str]:
        nonlocal joined
        if isinstance(node, list):
            result: set[str] = set()
            for item in node:
                result |= walk(item)
            return result
        if isinstance(node, str):
            return component_sources(node, rows)
        if not isinstance(node, dict):
            return set()
        own = component_sources(str(node.get('label') or node.get('caption') or ''), rows)
        children: set[str] = set()
        for key, value in node.items():
            if isinstance(value, (dict, list)) and key != 'edges':
                children |= walk(value)
        if len(own) > 1 or (own and children - own):
            joined = True
        result = own | children
        if node.get('id'):
            ids[str(node['id'])] = result
        # Explicit groups/nesting joins their members in one container.
        if node.get('members'):
            members = set().union(*(ids.get(str(i), set()) for i in node['members']))
            joined |= len(members) > 1
        return result
    sources = walk(spec.content)
    def edges(node: Any) -> None:
        nonlocal joined
        if isinstance(node, list):
            for item in node: edges(item)
        elif isinstance(node, dict):
            if node.get("members"):
                members = set().union(*(ids.get(str(i), set()) for i in node["members"]))
                own = component_sources(str(node.get("label") or ""), rows)
                joined |= len(members | own) > 1
            for edge in node.get('edges', []):
                if isinstance(edge, dict):
                    a = ids.get(str(edge.get('from') or edge.get('source')), set())
                    b = ids.get(str(edge.get('to') or edge.get('target')), set())
                    joined |= bool(a and b and len(a | b) > 1)
            for value in node.values():
                if isinstance(value, (dict, list)): edges(value)
    edges(spec.content)
    if spec.archetype in {'layered_stack', 'image_transform_strip', 'video_frame_strip', 'swimlane_sequence'}:
        joined |= len(sources) > 1
    if joined and not spec.synthesis_label:
        error('unlabeled_composition', 'joined components from different sources require synthesis_label')
    negative = re.compile(r"\b(?:no|not|never|without|lacks?|doesn't|cannot|can't)\b", re.I)
    for cell in spec.content.get('cells', []):
        if not isinstance(cell, dict) or cell.get('value') != 'no':
            continue
        row_key = str(cell.get('row') or '')
        row_label = next((str(r.get('label') or '') for r in spec.content.get('rows', []) if isinstance(r, dict) and r.get('id') == row_key), row_key)
        bound = component_sources(row_label, rows)
        notes = [str(r.get('contribution_note') or '') for r in rows if not bound or str(r.get('label') or r.get('source_id') or '') in bound]
        if not any(negative.search(note) for note in notes):
            error('unsupported_negative_cell', 'a no cell requires a bound negative; use not_reported for missing detail')
    return findings
