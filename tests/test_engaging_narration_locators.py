"""The engaging-narration generator must not let the hosts read acquisition
locators aloud, and its prompt must budget non-source color explicitly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from conference_pipeline.model_governance.llm_client import LLMResult

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts/presentations/generate_engaging_narration.py"
    spec = importlib.util.spec_from_file_location("generate_engaging_narration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_locator_references_flags_poster_talk_and_hash_phrases() -> None:
    module = _load()
    payload = {
        "slides": [
            {"slide_id": "sl_01", "two_host": [{"speaker": "A", "text": "This poster tackles the trap problem."}], "single": ""},
            {"slide_id": "sl_02", "two_host": [], "single": "See poster 63306 for the details."},
            {"slide_id": "sl_03", "two_host": [{"speaker": "A", "text": "The Peer-Preservation paper shows agents protecting peers."}], "single": "The poster session was packed."},
            {"slide_id": "sl_04", "two_host": [], "single": "The recording talk_media_001_9f4274a876 covers planning."},
        ]
    }
    hits = module.locator_references(payload)
    assert [h.split("]")[0].strip("[") for h in hits] == ["sl_01", "sl_02", "sl_04"]


def test_prompt_rules_name_works_and_budget_color() -> None:
    module = _load()
    spec = module.build_style_spec("Agentic AI", "ICML 2026")
    assert "NAME WORKS, NEVER LOCATORS" in spec
    assert "A poster IS a paper" in spec
    assert "COLOR BUDGET" in spec
    assert "never two in a row" in spec
    assert '"hook_kind"' in module.OUTPUT_INSTRUCTIONS
    assert '"aside"' in module.OUTPUT_INSTRUCTIONS


def test_a_reply_that_used_the_whole_ceiling_stops_with_a_reason_and_keeps_the_text(monkeypatch, tmp_path) -> None:
    module = _load()
    cut_off = LLMResult(
        text='{"slides": [{"slide_id": "sl_01"', stop_reason="max_tokens", provider="anthropic",
        model="claude-opus-5-5", block_types=("thinking", "text"), output_tokens=48000,
    )
    monkeypatch.setattr(module, "complete", lambda *args, **kwargs: cut_off)
    monkeypatch.setattr(module, "has_credentials", lambda provider: True)
    kept = tmp_path / "truncated.txt"
    with pytest.raises(module.NarrationTruncated) as raised:
        module.call_model("prompt", save_truncated_to=kept)
    message = str(raised.value)
    assert "whole" in message and "thinking, text" in message and "REASONING_EFFORT" in message
    assert kept.read_text(encoding="utf-8") == cut_off.text


def test_narration_asks_for_low_reasoning_effort_by_default(monkeypatch) -> None:
    module = _load()
    seen: dict = {}

    def fake_complete(*args, **kwargs):
        seen.update(kwargs)
        return LLMResult(text="{}", stop_reason="end", provider="anthropic", model="m")

    monkeypatch.setattr(module, "complete", fake_complete)
    monkeypatch.setattr(module, "has_credentials", lambda provider: True)
    module.call_model("prompt")
    assert seen["reasoning_effort"] == "low"
    assert seen["max_tokens"] >= 48000


def _aside_payload(text: str, *, aside: bool = True) -> dict:
    return {"slides": [{"slide_id": "sl_05", "aside": aside, "two_host": [{"speaker": "A", "text": text}], "single": text}]}


def test_asides_naming_people_projects_or_years_the_deck_cannot_back_are_flagged() -> None:
    module = _load()
    prompt = "Claim: residual flows. Source: Physics-Informed Residual Flows"
    for text, expected in (
        ("Quick aside -- Foldit players folded proteins better than the lab's software.", "Foldit"),
        ("This reminds me of the xkcd comic about jelly beans and false positives.", "comic"),
        ("Edward Lorenz found it in 1961 when a rounding error changed the weather run.", "1961"),
        ("It echoes how Thomas Hales checked the Kepler conjecture.", "Hales"),
    ):
        hits = module.aside_specifics(_aside_payload(text), prompt, hosts=("Iapetus", "Erinome"))
        assert hits and expected in hits[0], (text, hits)


def test_a_plain_aside_and_names_the_deck_itself_supplies_are_not_flagged() -> None:
    module = _load()
    prompt = "Source: Physics-Informed Residual Flows; Saga evolves objectives"
    plain = "Quick aside -- imagine a student who only memorises past exams and then meets a new question."
    assert module.aside_specifics(_aside_payload(plain), prompt) == []
    backed = "This reminds me of how Saga moves the goalposts, and Iapetus would say the same."
    assert module.aside_specifics(_aside_payload(backed), prompt, hosts=("Iapetus",)) == []
    # Only slides marked as asides are checked; a source-backed slide may name anything the sources do.
    assert module.aside_specifics(_aside_payload("Edward Lorenz in 1961", aside=False), prompt) == []


def test_the_prompt_no_longer_invites_historical_anecdotes_or_claims_every_source_is_shown() -> None:
    module = _load()
    spec = module.build_style_spec("AI for science", "ICML 2026")
    assert "well-known episode from the history" not in spec
    assert "NO SPECIFICS" in spec
    closing = module.build_prompt(
        [{"slide_number": 26, "slide_id": "s-resources", "kind": "resources"}], {}, audience_title="x",
    )
    assert "the main sources are on screen" in closing
    assert "Never say that every source" in closing
