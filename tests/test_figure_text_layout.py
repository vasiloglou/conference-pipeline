"""The slide-text layout: `key_points`, its builder, and its quota exemption."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from conference_pipeline.figures.compiler import compile_figure
from conference_pipeline.figures.errors import FigureSpecError
from conference_pipeline.figures.examples import EXAMPLES
from conference_pipeline.figures.grounding import STRONG, grounding_level, grounding_text
from conference_pipeline.figures.selection import deck_variety_report, select_archetype
from conference_pipeline.figures.spec import ARCHETYPES, FAMILY_TEXT, QUOTA_EXEMPT_FAMILIES, load_spec, validate_spec
from conference_pipeline.figures.text_layout import drawn_strings, first_sentence, key_points_spec, ungrounded_strings

SLIDE = {
    "slide_id": "sl_03_tools-carry-the-science",
    "section_id": "S1_from_predictors_to_loops",
    "title": "Tools carry the science: Rosetta and simulators, not chat",
    "claims": [
        "In Agent Rosetta the LLM orchestrates while design software supplies the constraint. "
        "Every split adds steps a single agent does not pay for."
    ],
    "visible_text": [
        "LLM plans; the domain tool evaluates",
        "Agent Rosetta handles non-canonical residues",
        "Tool use is not a standard baseline vs validation",
    ],
}


def _load_patch_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "patch_figure_spec.py"
    spec = importlib.util.spec_from_file_location("patch_figure_spec_text", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_key_points_is_a_quota_exempt_text_archetype() -> None:
    assert ARCHETYPES["key_points"].family == FAMILY_TEXT
    assert FAMILY_TEXT in QUOTA_EXEMPT_FAMILIES


def test_the_builder_carries_only_the_slides_own_words() -> None:
    payload = key_points_spec(SLIDE)
    assert payload["archetype"] == "key_points"
    assert payload["title"] == SLIDE["title"]
    assert payload["claim"] == "In Agent Rosetta the LLM orchestrates while design software supplies the constraint."
    assert payload["takeaway"] == ""
    assert [p["text"] for p in payload["content"]["points"]] == SLIDE["visible_text"]
    assert payload["section"] == SLIDE["section_id"]
    assert payload["figure_id"] == f"{SLIDE['slide_id']}-figure"
    compile_figure(payload)


def test_every_drawn_string_passes_the_grounding_rule() -> None:
    """Verbatim text is named outright, so grounding passes trivially -- even for
    a line a component filter would treat as a comparator foil ('baseline vs')."""

    payload = key_points_spec(SLIDE)
    pool = grounding_text(SLIDE)
    assert all(grounding_level(text, pool)[0] == STRONG for text in drawn_strings(payload))
    assert ungrounded_strings(payload, SLIDE) == []


def test_an_invented_line_is_caught_by_the_grounding_check() -> None:
    payload = key_points_spec(SLIDE)
    payload["content"]["points"].append({"text": "Hand-designed harness outperforms"})
    payload["takeaway"] = "Simulators beat chat"
    assert ungrounded_strings(payload, SLIDE) == ["Simulators beat chat", "Hand-designed harness outperforms"]


@pytest.mark.parametrize("missing", ["visible_text", "claims"])
def test_a_slide_without_text_or_claim_is_refused(missing: str) -> None:
    slide = {k: v for k, v in SLIDE.items() if k != missing}
    with pytest.raises(FigureSpecError):
        key_points_spec(slide)


def test_first_sentence_is_verbatim() -> None:
    assert first_sentence("One thing. Two things.") == "One thing."
    assert first_sentence("Only one sentence") == "Only one sentence"


def test_key_points_accepts_no_takeaway_and_rejects_extra_point_fields() -> None:
    payload = copy.deepcopy(EXAMPLES["key_points"])
    assert validate_spec(load_spec(payload, strict=False)) == []
    payload["content"]["points"][0]["id"] = "p1"
    assert any("carries only text" in e for e in validate_spec(load_spec(payload, strict=False)))
    # Every other archetype still needs a takeaway.
    flow = copy.deepcopy(EXAMPLES["flow"])
    flow["takeaway"] = ""
    assert "takeaway is required" in validate_spec(load_spec(flow, strict=False))


def test_text_layout_slides_are_left_out_of_the_variety_quotas() -> None:
    mixed = ["flow", "feature_matrix", "bar_compare", "loop", "stat_callout"]
    with_text = mixed[:2] + ["key_points"] * 3 + mixed[2:]
    base = deck_variety_report(mixed)
    report = deck_variety_report(with_text, sections=["a"] * len(with_text))
    assert report.findings == base.findings
    assert report.total == base.total
    assert report.family_counts == base.family_counts
    assert report.exempt_counts == {"key_points": 3}
    assert FAMILY_TEXT not in report.section_counts["a"]
    assert report.to_dict()["exempt_from_quotas"] == {"key_points": 3}
    # A deck of only text layouts has no figure mix to judge.
    assert deck_variety_report(["key_points"] * 6).passed


def test_selection_never_picks_the_text_layout() -> None:
    history = ["key_points"] * 10
    outcome = select_archetype(slide_id="sl_x", visual_role="mechanism: pipeline", history=history)
    assert outcome.archetype != "key_points"


def _deck(tmp_path: Path) -> Path:
    deck = tmp_path / "deck"
    (deck / "figure_specs").mkdir(parents=True)
    payload = copy.deepcopy(EXAMPLES["flow"])
    payload.update(figure_id=f"{SLIDE['slide_id']}-figure", slide_id=SLIDE["slide_id"], section="S1_from_predictors_to_loops")
    (deck / "figure_specs" / f"{SLIDE['slide_id']}-figure.json").write_text(json.dumps(payload), encoding="utf-8")
    (deck / "presentation_deck.json").write_text(json.dumps({"slides": [SLIDE]}), encoding="utf-8")
    return deck


def test_patch_replace_text_layout_builds_the_spec_from_the_contract(tmp_path: Path) -> None:
    module = _load_patch_script()
    deck = _deck(tmp_path)
    rc = module.main(["--deck-dir", str(deck), "--slide-id", "sl_03", "--outcome", "replace",
                      "--text-layout", "--by", "tester", "--reason", "no honest picture"])
    assert rc == 0
    spec = json.loads((deck / "figure_specs" / f"{SLIDE['slide_id']}-figure.json").read_text(encoding="utf-8"))
    assert spec["archetype"] == "key_points"
    assert spec["slide_id"] == SLIDE["slide_id"]
    assert [p["text"] for p in spec["content"]["points"]] == SLIDE["visible_text"]
    history = json.loads((deck / "figure_specs" / f"{SLIDE['slide_id']}-figure.patch.json").read_text(encoding="utf-8"))
    assert history[0]["outcome"] == "replace" and history[0]["text_layout"] is True


def test_patch_text_layout_flag_combinations_are_refused(tmp_path: Path) -> None:
    module = _load_patch_script()
    deck = _deck(tmp_path)
    other = tmp_path / "other.json"
    other.write_text(json.dumps(EXAMPLES["flow"]), encoding="utf-8")
    common = ["--deck-dir", str(deck), "--slide-id", "sl_03", "--by", "t", "--reason", "r"]
    with pytest.raises(SystemExit):
        module.main([*common, "--outcome", "replace", "--text-layout", "--spec", str(other)])
    with pytest.raises(SystemExit):
        module.main([*common, "--outcome", "accept", "--text-layout"])


def test_patch_text_layout_refuses_a_slide_with_no_text(tmp_path: Path) -> None:
    module = _load_patch_script()
    deck = _deck(tmp_path)
    (deck / "presentation_deck.json").write_text(
        json.dumps({"slides": [{**SLIDE, "visible_text": []}]}), encoding="utf-8")
    before = (deck / "figure_specs" / f"{SLIDE['slide_id']}-figure.json").read_text(encoding="utf-8")
    rc = module.main(["--deck-dir", str(deck), "--slide-id", "sl_03", "--outcome", "replace",
                      "--text-layout", "--by", "t", "--reason", "r"])
    assert rc == 1
    assert (deck / "figure_specs" / f"{SLIDE['slide_id']}-figure.json").read_text(encoding="utf-8") == before
