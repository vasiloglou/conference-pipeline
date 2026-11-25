"""A fuzz corpus for every archetype (design spec section 9).

The spec asks for "unit tests per archetype over a fuzz corpus: 1-12 nodes,
labels 4-60 chars, unicode and math, RTL-safe wrapping, every size class; the
geometry gate must report zero findings for every generated case." Only `flow`
and `bar_compare` ever had one, which is why real authored specs kept finding
layout defects that thirty-five worked examples never did: the examples were
written by the same person who wrote the layouts, with the counts and label
lengths that layout happened to handle.

Two kinds of case are generated:

* **Structural** -- hand-built payloads for the archetypes whose SHAPE varies
  (trees fan out and nest, graphs have skip edges and orphans, strips have
  links, scenes have collisions). These are where every real refusal lives.
* **Label-substituted** -- every worked example re-labelled from the awkward
  label set, which covers wrapping, shrinking and box growth for the other
  archetypes without inventing structures their layouts never promised.

A generated payload that fails `validate_spec` is dropped rather than tested:
the corpus exercises the COMPILER, and a spec the validator rejects never
reaches it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterator

from conference_pipeline.figures.examples import EXAMPLES
from conference_pipeline.figures.spec import ARCHETYPES, SPEC_VERSION, load_spec, validate_spec

# One character to sixty, unicode, math notation, right-to-left scripts, CJK
# with nothing to wrap on, and a token no column can hold. Kept identical to
# the list `test_figure_compiler.py` already fuzzes `flow` with, so the two
# suites agree on what "an awkward label" means.
LABELS: list[str] = [
    "A",
    "Encoder",
    "Vision encoder",
    "Concept-aware batch sampling for contrastive training",
    "Qwen 3VL",
    "supercalifragilisticexpialidocious",
    "Édouard Grave & co-authors",
    "3-D Gaussian splatting",
    "an unusually long label that runs to sixty characters or so ok",
    "σ = 0.5 · ∇f(x) ⊗ Wᵀ",
    "الرؤية الحاسوبية",
    "ראייה ממוחשבת ועיבוד",
    "視覚言語モデル",
]

MAX_CASES = 1500


@dataclass(frozen=True)
class FuzzCase:
    """One generated spec plus the labels it expects the figure to carry."""

    case_id: str
    archetype: str
    payload: dict[str, Any]


def _label(index: int) -> str:
    return LABELS[index % len(LABELS)]


def _envelope(archetype: str, content: dict[str, Any], **extra: Any) -> dict[str, Any]:
    grounding = ARCHETYPES[archetype].grounding_kinds
    kind = "schematic" if "schematic" in grounding else "data"
    payload: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "figure_id": f"fuzz-{archetype}",
        "slide_id": "sl_fuzz",
        "archetype": archetype,
        "claim": "A fuzzed claim about a mechanism.",
        "takeaway": "Fuzzed",
        "grounding": {"kind": kind, "ledger_rows": [] if kind == "schematic" else ["nc-0001"]},
        "content": content,
    }
    payload.update(extra)
    return payload


# -- structural generators -----------------------------------------------------

def _tree(archetype: str, fan: int, depth: int, label_index: int) -> dict[str, Any]:
    counter = [0]

    def children(level: int) -> list[dict[str, Any]]:
        if level > depth:
            return []
        out = []
        for i in range(fan):
            counter[0] += 1
            node: dict[str, Any] = {
                "id": f"n{counter[0]}",
                "label": _label(label_index + counter[0]),
                "kind": "process" if level < depth else "output",
            }
            kids = children(level + 1)
            if kids:
                node["children"] = kids
            if archetype == "decision_tree":
                node["branch_label"] = "yes" if i % 2 == 0 else "no"
            out.append(node)
        return out

    root_kind = "decision" if archetype == "decision_tree" else "model"
    return _envelope(archetype, {
        "root": {"id": "root", "label": _label(label_index), "kind": root_kind},
        "children": children(1),
    })


def _graph(archetype: str, nodes: int, label_index: int, *, orphan: bool, skip: bool,
           groups: int = 0, edge_labels: bool = True) -> dict[str, Any]:
    node_list = [
        {"id": f"n{i}", "label": _label(label_index + i), "kind": "process"}
        for i in range(nodes)
    ]
    edges = [
        {"from": f"n{i}", "to": f"n{i + 1}", **({"label": _label(i + 3)} if edge_labels else {})}
        for i in range(nodes - 1)
    ]
    if skip and nodes >= 3:
        edges.append({"from": "n0", "to": f"n{nodes - 1}", "label": "skips"})
    if orphan:
        node_list.append({"id": "loose", "label": _label(label_index + 7), "kind": "data"})
    content: dict[str, Any] = {"nodes": node_list, "edges": edges}
    if groups:
        members = [[] for _ in range(groups)]
        for index, node in enumerate(node_list):
            members[index % groups].append(node["id"])
        content["groups"] = [
            {"id": f"g{g}", "label": _label(label_index + g + 1), "members": members[g]}
            for g in range(groups) if members[g]
        ]
    return _envelope(archetype, content)


def _loop(nodes: int, label_index: int, *, edge_labels: bool, output: bool) -> dict[str, Any]:
    content: dict[str, Any] = {
        "nodes": [
            {
                "id": f"n{i}", "label": _label(label_index + i), "kind": "process",
                **({"edge_label": _label(i + 2)} if edge_labels else {}),
            }
            for i in range(nodes)
        ]
    }
    if output:
        content["centre_label"] = _label(label_index + 9)
    return _envelope("loop", content)


def _scatter(points: int, label_index: int) -> dict[str, Any]:
    return _envelope("scatter_frontier", {
        "x": {"label": _label(label_index), "unit": "%"},
        "y": {"label": _label(label_index + 1), "unit": "x"},
        "frontier": points >= 2,
        "points": [
            {"id": f"p{i}", "label": _label(label_index + i), "x": 10.0 + i * 7.0,
             "y": 1.0 + (i % 3), "on_frontier": True}
            for i in range(points)
        ],
    })


def _spatial(count: int, label_index: int, *, clustered: bool) -> dict[str, Any]:
    kinds = ["box", "camera", "splats", "point_cloud", "path"]
    return _envelope("spatial_3d_schematic", {
        "primitives": [
            {
                "id": f"p{i}", "label": _label(label_index + i),
                "kind": kinds[i % len(kinds)],
                "position": {"col": 0 if clustered else (i - count / 2) * 1.2,
                             "row": 0 if clustered else (i % 3) - 1,
                             "level": 0 if clustered else 0.4 * (i % 2)},
            }
            for i in range(count)
        ],
    })


def _strip(archetype: str, count: int, label_index: int, *, links: bool) -> dict[str, Any]:
    key = "frames" if archetype == "video_frame_strip" else "panels"
    content: dict[str, Any] = {
        key: [
            {"id": f"f{i}", "caption": _label(label_index + i)}
            for i in range(count)
        ]
    }
    if archetype == "video_frame_strip":
        content["time_label"] = "frame order"
        if links and count >= 2:
            content["links"] = [
                {"from": i, "to": i + 1, "label": _label(i + 4)} for i in range(count - 1)
            ]
    elif links:
        for panel in content[key][1:]:
            panel["via"] = _label(label_index + 5)
    return _envelope(archetype, content)




def _fusion(count: int, label_index: int, *, missing: bool, alternatives: int,
            outputs: int) -> dict[str, Any]:
    """Modalities into one mechanism, several results, or competing mechanisms."""

    modalities = [
        {"id": f"m{i}", "glyph": "photo", "label": _label(label_index + i),
         **({"missing": True} if missing and i == count - 1 else {})}
        for i in range(count)
    ]
    fusion: dict[str, Any] = {"label": _label(label_index + count)}
    if alternatives:
        fusion["alternatives"] = [
            {"id": f"alt{i}", "label": _label(label_index + count + 1 + i),
             "output": {"label": _label(label_index + count + 4 + i)}}
            for i in range(alternatives)
        ]
    elif outputs == 1:
        fusion["output"] = {"label": _label(label_index + count + 1)}
    else:
        fusion["output"] = [{"label": _label(label_index + count + 1 + i)} for i in range(outputs)]
    return _envelope("modality_fusion", {"modalities": modalities, "fusion": fusion},
                     emphasis={"highlight": ["alt0"]} if alternatives else {})


def _structural_cases() -> Iterator[FuzzCase]:
    for label_index in range(len(LABELS)):
        for archetype in ("hierarchy_tree", "decision_tree"):
            for fan in (1, 2, 4, 7):
                for depth in (1, 2):
                    yield FuzzCase(
                        f"{archetype}-fan{fan}-d{depth}-l{label_index:02d}",
                        archetype, _tree(archetype, fan, depth, label_index),
                    )
        for nodes in (2, 3, 5, 8):
            for orphan in (False, True):
                for skip in (False, True):
                    yield FuzzCase(
                        f"flow-n{nodes}-o{int(orphan)}-s{int(skip)}-l{label_index:02d}",
                        "flow", _graph("flow", nodes, label_index, orphan=orphan, skip=skip),
                    )
        for nodes in (2, 4, 6, 9):
            for groups in (1, 2, 3):
                yield FuzzCase(
                    f"nested-n{nodes}-g{groups}-l{label_index:02d}", "nested_architecture",
                    _graph("nested_architecture", nodes, label_index, orphan=False,
                           skip=nodes >= 4, groups=groups),
                )
        for nodes in (2, 3, 5, 7):
            for edge_labels in (False, True):
                yield FuzzCase(
                    f"loop-n{nodes}-e{int(edge_labels)}-l{label_index:02d}", "loop",
                    _loop(nodes, label_index, edge_labels=edge_labels, output=True),
                )
        for points in (1, 2, 3, 6, 10):
            yield FuzzCase(f"scatter-p{points}-l{label_index:02d}", "scatter_frontier",
                           _scatter(points, label_index))
        for count in (1, 3, 5, 7):
            for clustered in (False, True):
                yield FuzzCase(
                    f"spatial-n{count}-c{int(clustered)}-l{label_index:02d}",
                    "spatial_3d_schematic", _spatial(count, label_index, clustered=clustered),
                )
        for archetype in ("video_frame_strip", "image_transform_strip"):
            for count in (2, 3, 5):
                for links in (False, True):
                    yield FuzzCase(
                        f"{archetype}-n{count}-k{int(links)}-l{label_index:02d}",
                        archetype, _strip(archetype, count, label_index, links=links),
                    )
        for count in (2, 4):
            for missing in (False, True):
                for alternatives, outputs in ((0, 1), (0, 3), (2, 1), (3, 1)):
                    yield FuzzCase(
                        f"fusion-m{count}-x{int(missing)}-a{alternatives}-o{outputs}-l{label_index:02d}",
                        "modality_fusion",
                        _fusion(count, label_index, missing=missing,
                                alternatives=alternatives, outputs=outputs),
                    )


# -- label substitution over the worked examples -------------------------------

_LABEL_FIELDS = ("label", "caption", "title", "text", "name", "sublabel", "edge_label")


def _relabel(node: Any, offset: int, counter: list[int], renames: dict[str, str]) -> Any:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in _LABEL_FIELDS and isinstance(value, str) and value.strip():
                counter[0] += 1
                replacement = _label(offset + counter[0])
                renames.setdefault(value, replacement)
                out[key] = replacement
            else:
                out[key] = _relabel(value, offset, counter, renames)
        return out
    if isinstance(node, list):
        return [_relabel(item, offset, counter, renames) for item in node]
    return node


_REFERENCE_FIELDS = ("row", "col", "series", "from", "to", "members", "highlight")


def _rewrite_references(node: Any, renames: dict[str, str]) -> Any:
    """Follow a rename through the keys that point AT a label.

    `feature_matrix` and `heatmap_grid` address their cells by row and column
    LABEL, so renaming a row without renaming its cells produces a spec the
    validator rejects -- and the archetype silently vanishes from the corpus.
    """

    if isinstance(node, dict):
        return {
            key: (renames.get(value, value)
                  if key in _REFERENCE_FIELDS and isinstance(value, str)
                  else _rewrite_references(value, renames))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_references(item, renames) for item in node]
    return node


def _relabelled_cases() -> Iterator[FuzzCase]:
    for archetype in sorted(EXAMPLES):
        for offset in range(len(LABELS)):
            payload = copy.deepcopy(EXAMPLES[archetype])
            renames: dict[str, str] = {}
            payload["content"] = _relabel(payload["content"], offset, [0], renames)
            payload["content"] = _rewrite_references(payload["content"], renames)
            payload["figure_id"] = f"fuzz-{archetype}-{offset}"
            yield FuzzCase(f"{archetype}-relabel{offset:02d}", archetype, payload)


def build_corpus() -> list[FuzzCase]:
    """Every generated case whose payload the validator accepts, in a fixed order."""

    seen: set[str] = set()
    cases: list[FuzzCase] = []
    for case in (*_structural_cases(), *_relabelled_cases()):
        if case.case_id in seen:
            continue
        try:
            spec = load_spec(case.payload, strict=False)
        except Exception:
            continue
        if validate_spec(spec):
            continue  # not a compiler question: the validator already refuses it
        seen.add(case.case_id)
        cases.append(case)
    return cases


CORPUS = build_corpus()
