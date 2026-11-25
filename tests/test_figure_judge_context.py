"""The judge sees pixels and a sequence number, and nothing else.

Three rounds of findings were shaped by that. It read the compiler's mandatory
honesty label as "a production-note-like footer"; it could not tell a figure
that omits numbers from one that is FORBIDDEN to carry them; and with ten
check ids and no definition for any of them, unrelated complaints piled into
`generic_placeholder_visual` while every semantic objection landed in `other`.

The enum is deliberately unchanged, so counts stay comparable with the rounds
run before this. Only the context and the definitions are new, and the report
records which rubric produced it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.figures.judge_context import build_figure_context  # noqa: E402
from conference_pipeline.presentation.vision_qc import (  # noqa: E402
    CHECK_ID_GLOSSES,
    RUBRIC_VERSION_PLAIN,
    RUBRIC_VERSION_WITH_CONTEXT,
    _issue_severity,
    figure_context_block,
)


def _manifest(tmp_path: Path, figure_id: str, slide_id: str, archetype: str) -> Path:
    compiled = tmp_path / "assets" / "compiled"
    compiled.mkdir(parents=True, exist_ok=True)
    (compiled / f"{figure_id}.manifest.json").write_text(
        json.dumps({
            "figure_id": figure_id,
            "slide_id": slide_id,
            "archetype": archetype,
            "family": "topology",
            "claim": "A claim about a mechanism.",
            "takeaway": "A takeaway",
            "grounding": {"kind": "schematic"},
            "glyphs_used": ["photo"],
        }),
        encoding="utf-8",
    )
    return compiled


def test_a_filename_carrying_the_figure_id_beats_any_positional_guess(tmp_path: Path) -> None:
    compiled = _manifest(tmp_path, "fig-a", "sl_01_first", "flow")
    _manifest(tmp_path, "fig-b", "sl_02_second", "loop")
    shots = [tmp_path / "02__fig-a.png"]
    context = build_figure_context(compiled_dir=compiled, screenshots=shots)
    assert context["02__fig-a.png"]["archetype"] == "flow"
    assert context["02__fig-a.png"]["join_method"] == "figure_id"


def test_a_render_spec_maps_a_slide_number_onto_its_figure(tmp_path: Path) -> None:
    compiled = _manifest(tmp_path, "fig-a", "sl_01_first", "flow")
    _manifest(tmp_path, "fig-b", "sl_02_second", "loop")
    context = build_figure_context(
        compiled_dir=compiled,
        screenshots=[tmp_path / "slide_002.png"],
        render_spec={"slides": [{"slide_id": "sl_01_first"}, {"slide_id": "sl_02_second"}]},
    )
    assert context["slide_002.png"]["archetype"] == "loop"
    assert context["slide_002.png"]["join_method"] == "render_spec"


def test_the_positional_fallback_says_that_it_guessed(tmp_path: Path) -> None:
    """Correct for a directory rasterised in one pass, wrong once one is added."""

    compiled = _manifest(tmp_path, "fig-a", "sl_01_first", "flow")
    context = build_figure_context(compiled_dir=compiled, screenshots=[tmp_path / "slide-01.png"])
    assert context["slide-01.png"]["join_method"] == "positional"


def test_the_context_block_states_the_claim_and_the_conventions() -> None:
    block = figure_context_block({
        "claim": "Reflection re-queries the image.",
        "takeaway": "Reflection re-queries",
        "archetype": "loop",
        "family": "topology",
        "grounding_kind": "schematic",
        "glyphs_used": [],
    })
    assert "Reflection re-queries the image." in block
    assert "schematic" in block
    # The badge and the no-numbers rule are the two conventions it kept reporting.
    assert "required honesty" in block
    assert "carries NO numbers" in block


def test_every_check_id_has_a_definition() -> None:
    """They were an unglossed enum: ten names and not one definition."""

    for name, gloss in CHECK_ID_GLOSSES.items():
        assert gloss and len(gloss) > 20, name
    assert "does not depict the subject" in CHECK_ID_GLOSSES["generic_placeholder_visual"]


def test_prose_no_longer_promotes_an_advisory_finding_to_a_blocker() -> None:
    """The word "flow" in a description used to stop a deck."""

    issue = {
        "check_id": "other",
        "severity": "warn",
        "description": "The flow reads left to right, which matches the claim.",
    }
    assert _issue_severity(issue) == "blocker"
    assert _issue_severity(issue, enum_only=True) == "warn"


def test_the_enum_still_blocks_on_its_own_terms() -> None:
    issue = {"check_id": "text_overlap", "severity": "warn", "description": "labels overlap"}
    assert _issue_severity(issue, enum_only=True) == "blocker"


def test_the_two_rubrics_are_distinguishable() -> None:
    assert RUBRIC_VERSION_PLAIN != RUBRIC_VERSION_WITH_CONTEXT


def test_the_judge_cli_accepts_a_deck_to_take_context_from() -> None:
    """Without one it reviews pixels alone, exactly as it did before."""

    import importlib.util

    path = REPO_ROOT / "scripts" / "presentations" / "inspect_rendered_slides_llm.py"
    spec = importlib.util.spec_from_file_location("inspect_rendered_slides_llm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = path.read_text(encoding="utf-8")
    assert "--figure-context-deck-dir" in source
    assert "figure_context=figure_context" in source
