"""The executive narration stays inside a deck word budget under the 40-minute delivery gate."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "presentations"))

import generate_executive_narration as narration  # noqa: E402


def _fake_opus(words_per_target: float, calls: list):
    def call(client, model, briefs, system):
        calls.append([tw for _, _, tw in briefs])
        return {sn: " ".join(["word"] * int(tw * words_per_target)) for sn, _, tw in briefs}
    return call


def test_targets_are_scaled_so_a_long_deck_fits_the_budget(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(narration, "call_opus", _fake_opus(1.0, calls))
    briefs = [(index, "brief", 200) for index in range(1, 41)]  # 8,000 planned words
    narr = narration.narrate_within_budget(None, "m", briefs, "s", budget=5000)
    assert sum(len(text.split()) for text in narr.values()) <= 5000
    assert len(calls) == 1 and sum(calls[0]) < 5000


def test_an_overshooting_model_gets_one_tighter_retry_then_fails_closed(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(narration, "call_opus", _fake_opus(1.3, calls))
    briefs = [(index, "brief", 150) for index in range(1, 41)]
    narr = narration.narrate_within_budget(None, "m", briefs, "s", budget=5000)
    assert len(calls) == 2
    assert sum(len(text.split()) for text in narr.values()) <= 5000

    monkeypatch.setattr(narration, "call_opus", _fake_opus(6.0, []))
    with pytest.raises(SystemExit, match="over the 5000-word budget"):
        narration.narrate_within_budget(None, "m", briefs, "s", budget=5000)
