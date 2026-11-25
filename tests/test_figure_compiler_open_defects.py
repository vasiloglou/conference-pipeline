"""Real dense-layout regressions and explicit topology semantics."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from xml.etree import ElementTree

import pytest

from conference_pipeline.figures import tokens
from conference_pipeline.figures.compiler import _render_once, compile_figure
from conference_pipeline.figures.errors import FigureSpecError
from conference_pipeline.figures.examples import EXAMPLES
from conference_pipeline.figures.frame import draw_frame
from conference_pipeline.figures.layouts._common import LayoutOptions
from conference_pipeline.figures.layouts.topology import _graph_from_spec
from conference_pipeline.figures.ledger import NumericLedger
from conference_pipeline.figures.lint import lint_spec
from conference_pipeline.figures.spec import load_spec
from conference_pipeline.figures.svgdoc import SvgDoc
from conference_pipeline.presentation.svg_geometry import parse_scene

FIXTURES = Path(__file__).parent / "fixtures" / "figure_compiler_regressions"


def test_feature_matrix_key_sits_below_the_last_data_band() -> None:
    svg = compile_figure(EXAMPLES["feature_matrix"]).svg
    elements = list(ElementTree.fromstring(svg).iter())
    bands = [
        float(element.attrib["y"]) + float(element.attrib["height"])
        for element in elements
        if element.tag.endswith("rect")
        and element.attrib.get("fill") == tokens.PANEL
        and "y" in element.attrib
    ]
    key_marks = [
        float(element.attrib["cy"]) - float(element.attrib["r"])
        for element in elements
        if element.tag.endswith("circle")
        and float(element.attrib.get("r", "0")) <= 5.1
        and float(element.attrib.get("cy", "0")) > 500
    ]

    assert bands and key_marks
    assert min(key_marks) > max(bands) + 8


def test_quadrant_markers_with_nearby_qualitative_scores_do_not_overlap() -> None:
    payload = copy.deepcopy(EXAMPLES["quadrant_map"])
    payload["required_entities"] = []
    payload["content"]["points"] = [
        {"id": "a", "label": "Method A for open vocabulary transfer", "x": 0.78, "y": 0.24, "basis": "stated by the source"},
        {"id": "b", "label": "Method B for open vocabulary transfer", "x": 0.72, "y": 0.28, "basis": "stated by the source"},
        {"id": "c", "label": "Method C for open vocabulary transfer", "x": 0.80, "y": 0.20, "basis": "stated by the source"},
    ]
    svg = compile_figure(payload).svg
    markers = [
        (float(element.attrib["cx"]), float(element.attrib["cy"]), float(element.attrib["r"]))
        for element in ElementTree.fromstring(svg).iter()
        if element.tag.endswith("circle") and float(element.attrib.get("r", "0")) >= 7.0
    ]

    assert len(markers) == 3
    assert all(
        math.hypot(x1 - x2, y1 - y2) >= r1 + r2 + 6.0
        for index, (x1, y1, r1) in enumerate(markers)
        for x2, y2, r2 in markers[index + 1:]
    )


def test_linked_nested_subcolumns_leave_a_readable_horizontal_shaft() -> None:
    payload = json.loads((FIXTURES / "linked_subcolumns.json").read_text(encoding="utf-8"))
    scene = parse_scene(compile_figure(payload).svg)

    def node_box(label: str):
        word = next(text for text in scene.texts if text.body == label)
        return min(
            (shape for shape in scene.shapes
             if shape.kind == "rect" and shape.contains_point(word.x, word.y)),
            key=lambda shape: shape.width * shape.height,
        )

    source = node_box("Discrete")
    target = node_box("Multimodal")
    assert target.x0 - source.x1 >= 40.0


@pytest.mark.parametrize(
    ("name", "required_label"),
    [
        ("dense_nested_width", "Instrument/anatomy detection"),
        ("dense_nested_return", "tail-latency frames"),
        ("labeled_internal_route", "reason before act"),
        ("linked_group_wrapping", "vector field"),
    ],
)
def test_preserved_nested_architecture_cases_compile_without_losing_the_label(
    name: str, required_label: str,
) -> None:
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    figure = compile_figure(payload)

    assert figure.attempt == 0
    assert required_label in figure.manifest.labels_drawn
    assert required_label not in figure.manifest.labels_dropped


def _flow(nodes: list[dict], edges: list[dict]) -> dict:
    payload = copy.deepcopy(EXAMPLES["flow"])
    payload["content"] = {"nodes": nodes, "edges": edges}
    payload["required_entities"] = []
    return payload


def test_feedback_returns_in_a_separate_outer_lane() -> None:
    payload = _flow(
        [
            {"id": "capture", "label": "Capture", "kind": "input"},
            {"id": "reason", "label": "Reason", "kind": "process"},
            {"id": "act", "label": "Act", "kind": "output"},
        ],
        [
            {"from": "capture", "to": "reason"},
            {"from": "reason", "to": "act"},
            {"from": "act", "to": "capture", "kind": "feedback", "label": "update"},
        ],
    )
    figure = compile_figure(payload)
    doc = _render_once(load_spec(payload, mode="compile"), LayoutOptions(**figure.options))
    feedback = [edge for edge in doc.connectors if edge.stroke == tokens.ALT_VIOLET]

    assert figure.attempt <= 1
    assert "update" in figure.manifest.labels_drawn
    assert len(feedback) == 1
    route = feedback[0].points
    assert len(route) >= 4
    if figure.options.get("flip_direction"):
        assert route[0][1] > route[-1][1]
        assert route[1][0] == route[2][0]
        assert (route[1][0] < min(route[0][0], route[-1][0])
                or route[1][0] > max(route[0][0], route[-1][0]))
    else:
        assert route[0][0] > route[-1][0]
        assert route[1][1] == route[2][1]
        assert (route[1][1] < min(route[0][1], route[-1][1])
                or route[1][1] > max(route[0][1], route[-1][1]))


def test_feedback_without_a_forward_path_is_refused() -> None:
    payload = _flow(
        [
            {"id": "capture", "label": "Capture", "kind": "input"},
            {"id": "act", "label": "Act", "kind": "output"},
        ],
        [{"from": "act", "to": "capture", "kind": "feedback"}],
    )
    with pytest.raises(FigureSpecError, match="so the loop closes"):
        compile_figure(payload)


def test_parallel_methods_share_a_layer_and_merge_at_a_named_node() -> None:
    payload = _flow(
        [
            {"id": "a", "label": "Method A", "kind": "process"},
            {"id": "b", "label": "Method B", "kind": "process"},
            {"id": "join", "label": "Combined evidence", "kind": "merge"},
            {"id": "result", "label": "Result", "kind": "output"},
        ],
        [
            {"from": "a", "to": "b", "kind": "parallel", "label": "complementary"},
            {"from": "a", "to": "join"},
            {"from": "b", "to": "join"},
            {"from": "join", "to": "result"},
        ],
    )
    spec = load_spec(payload, mode="compile")
    frame = draw_frame(SvgDoc(), spec)
    graph, _, _ = _graph_from_spec(spec, frame.plot, LayoutOptions())
    by_id = graph.by_id()
    figure = compile_figure(payload)

    assert by_id["a"].layer == by_id["b"].layer < by_id["join"].layer
    assert figure.attempt == 0
    assert "complementary" in figure.manifest.labels_drawn
    assert 'stroke-dasharray="4 6"' in figure.svg


def test_merge_requires_two_actual_inputs_and_may_be_the_result() -> None:
    payload = _flow(
        [
            {"id": "a", "label": "Method A"},
            {"id": "join", "label": "Combined", "kind": "merge"},
        ],
        [{"from": "a", "to": "join"}],
    )
    with pytest.raises(FigureSpecError, match="needs at least two incoming"):
        compile_figure(payload)
    payload["content"]["nodes"].append({"id": "b", "label": "Method B"})
    payload["content"]["edges"].append({"from": "b", "to": "join"})
    assert compile_figure(payload).manifest.labels_dropped == []


def test_generic_flow_reports_implicit_merge_and_return() -> None:
    payload = _flow(
        [
            {"id": "a", "label": "Method A"},
            {"id": "b", "label": "Method B"},
            {"id": "join", "label": "Combined evidence"},
        ],
        [
            {"from": "a", "to": "join"},
            {"from": "b", "to": "join"},
            {"from": "join", "to": "a"},
        ],
    )
    findings = lint_spec(load_spec(payload, mode="compile"))

    assert {finding.code for finding in findings} >= {
        "flow_implicit_merge", "flow_implicit_feedback",
    }


def test_frame_strip_keeps_captions_inside_eight_narrow_panels() -> None:
    # The original paid-authored failing spec was regenerated. Preserve its
    # reported caption and panel count as a deterministic geometry guard.
    payload = copy.deepcopy(EXAMPLES["video_frame_strip"])
    payload["content"]["frames"] = [
        {"id": f"f{i}", "caption": "early frame establishing" if i == 0 else f"next frame, detail {i}"}
        for i in range(8)
    ]
    payload["content"]["frames"][1]["depicts"] = "early frame establishing"
    payload["content"]["links"] = []
    figure = compile_figure(payload)

    assert figure.attempt == 0
    assert "early frame establishing" in figure.manifest.labels_drawn


def _authoring_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "generate_figure_specs.py"
    spec = importlib.util.spec_from_file_location("figure_spec_author_for_topology_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flow_authoring_prompt_explains_the_new_relationships() -> None:
    module = _authoring_module()
    prompt = module.build_prompt(
        slide={"slide_id": "s1", "display_title": "Two methods combine"},
        archetype="flow", ledger=NumericLedger(topic_id="test"), ledger_hits=[],
    )

    assert 'kind: "feedback"' in prompt
    assert 'kind: "parallel"' in prompt
    assert 'kind: "merge"' in prompt
    critique = module.build_critic_prompt(
        payload=_flow([], []), slide={"title": "Two methods combine"}, archetype="flow",
    )
    assert "A claimed loop needs both its forward path" in critique
    assert "A claimed merge needs a named `merge` node" in critique


def test_a_retry_preserves_the_refused_frame_spec(tmp_path: Path) -> None:
    module = _authoring_module()
    path = tmp_path / "sl_18-figure.json"
    refused = copy.deepcopy(EXAMPLES["video_frame_strip"])
    refused["content"]["frames"] = []
    path.write_text(json.dumps(refused, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replacement = copy.deepcopy(EXAMPLES["video_frame_strip"])

    module._write_spec_preserving_refusal(path, replacement)

    history = list((tmp_path / "refused").glob("sl_18-figure.*.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text(encoding="utf-8")) == refused
    assert json.loads(path.read_text(encoding="utf-8")) == replacement
    module._write_spec_preserving_refusal(path, replacement)
    assert len(list((tmp_path / "refused").glob("*.json"))) == 1
