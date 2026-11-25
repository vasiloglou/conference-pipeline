from __future__ import annotations

from conference_pipeline.presentation.blueprint import drop_unauthored_sections
from conference_pipeline.presentation.blueprint_markdown import render_blueprint_markdown


def _blueprint() -> dict:
    return {
        "telescopic_section_map": [
            {"section_id": "S1", "title": "Loops", "slide_ids": ["sl_02"]},
            {"section_id": "S5", "title": "Molecules", "slide_ids": []},
        ],
        "subtopic_clusters": [
            {"cluster_id": "loops", "title": "Loops", "sources": ["SRC-001"], "slide_bindings": ["sl_02"]},
            {"cluster_id": "molecules", "title": "Molecules", "sources": ["SRC-009"], "slide_bindings": [],
             "source_ids": ["paper_abstract:m"]},
        ],
        "cluster_coverage_justification": {
            "covered_cluster_ids": ["loops", "molecules"], "required": ["loops", "molecules"], "reason": "r",
        },
        "slides": [{"slide_id": "sl_01", "title": "Title"}, {"slide_id": "sl_02", "title": "Loops", "section_id": "S1"}],
    }


def test_an_empty_section_and_its_unbound_cluster_are_dropped_and_recorded() -> None:
    blueprint = _blueprint()
    records = drop_unauthored_sections(blueprint, reasons={"S5": "authoring failed: model returned no text"})
    assert [s["section_id"] for s in blueprint["telescopic_section_map"]] == ["S1"]
    assert [c["cluster_id"] for c in blueprint["subtopic_clusters"]] == ["loops"]
    assert blueprint["cluster_coverage_justification"]["required"] == ["loops"]
    assert blueprint["cluster_coverage_justification"]["covered_cluster_ids"] == ["loops"]
    assert {"section_id": "S5", "title": "Molecules", "reason": "authoring failed: model returned no text"} in records
    assert any(r.get("cluster_id") == "molecules" and r["source_ids"] == ["paper_abstract:m"] for r in records)
    assert blueprint["dropped_sections"] == records


def test_nothing_is_dropped_from_a_complete_blueprint() -> None:
    blueprint = _blueprint()
    blueprint["telescopic_section_map"].pop()
    blueprint["subtopic_clusters"].pop()
    assert drop_unauthored_sections(blueprint) == []
    assert "dropped_sections" not in blueprint


def test_the_review_view_names_dropped_sections() -> None:
    blueprint = _blueprint()
    drop_unauthored_sections(blueprint, reasons={"S5": "authoring failed: refusal"})
    md = render_blueprint_markdown(blueprint, topic_title="Demo", stats={"slides": 1, "sources": 1, "works": 1, "people": 0})
    assert "**Dropped sections**" in md and "- Molecules: authoring failed: refusal" in md
