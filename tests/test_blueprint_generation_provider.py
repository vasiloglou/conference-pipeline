"""Blueprint generation is a governed task, not an Anthropic stage.

The live path goes through the shared model client, so any provider the task
allows for the chosen model is used, with that provider's credential. Nothing
here calls a provider.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline.presentation import blueprint_generation as bg

K3 = "openrouter/moonshotai/kimi-k3"


def _config(tmp_path: Path, **overrides) -> bg.BlueprintGenerationConfig:
    return bg.BlueprintGenerationConfig(
        workspace=tmp_path, presentation_id="deck", presentation_type="topic", **overrides
    )


def _result(text: str, stop_reason: str = "end"):
    return SimpleNamespace(text=text, stop_reason=stop_reason, block_types=("text",), receipt=None)


def test_a_kimi_blueprint_is_routed_to_openrouter_with_its_own_credential(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen: dict = {}

    def fake_complete(task_id, **kwargs):
        seen.update(kwargs, task_id=task_id)
        return _result('{"slides": []}')

    monkeypatch.setattr(bg, "complete", fake_complete)
    payload, _paths, _cost = bg._live_generate_blueprint(
        config=_config(tmp_path), workspace=tmp_path, provider="openrouter", model_id=K3, prompt="p",
    )
    assert payload == {"slides": []}
    assert (seen["task_id"], seen["provider"], seen["model"]) == (
        bg.BLUEPRINT_GENERATION_TASK_ID, "openrouter", K3,
    )
    assert seen["stream"] is True and seen["max_tokens"] == 24000


def test_a_missing_credential_for_the_chosen_provider_is_the_one_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present-but-irrelevant")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(bg.BlueprintGenerationError) as error:
        bg._live_generate_blueprint(
            config=_config(tmp_path), workspace=tmp_path, provider="openrouter", model_id=K3, prompt="p",
        )
    assert error.value.code == bg.BLUEPRINT_PROVIDER_CREDENTIALS_MISSING
    assert "OPENROUTER_API_KEY" in str(error.value)


def test_a_reply_cut_off_at_the_ceiling_is_a_failure_not_a_parse_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(bg, "complete", lambda task_id, **kw: _result('{"slides": [', stop_reason="max_tokens"))
    with pytest.raises(bg.BlueprintGenerationError, match="cut off"):
        bg._live_generate_blueprint(
            config=_config(tmp_path), workspace=tmp_path, provider="openrouter", model_id=K3, prompt="p",
        )


def test_the_config_model_alone_selects_the_provider() -> None:
    from conference_pipeline.model_governance.llm_client import resolve_selection

    assert resolve_selection(bg.BLUEPRINT_GENERATION_TASK_ID, None, K3) == ("openrouter", K3)
    assert resolve_selection(bg.BLUEPRINT_GENERATION_TASK_ID)[0] == "anthropic"
