"""Claude Fable rejects a custom temperature (found by the v6.1 Phase 3 adjudicator canary, 2026-11-18)."""

from conference_pipeline.model_governance.model_families import anthropic_rejects_temperature


def test_every_fable_model_rejects_a_custom_temperature():
    assert anthropic_rejects_temperature("claude-fable-5-2")
    assert anthropic_rejects_temperature("claude-fable-6")
    assert anthropic_rejects_temperature("claude-opus-5-5") and not anthropic_rejects_temperature("claude-haiku-4-5")


def test_haiku_5_5_rejects_temperature_and_haiku_4_5_does_not() -> None:
    # v6.1 Phase 5 pilot, 2026-11-21: HTTP 400 "`temperature` is deprecated for this model"
    assert anthropic_rejects_temperature("claude-haiku-5-5")
    assert not anthropic_rejects_temperature("claude-haiku-4-5-20251001")
