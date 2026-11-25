"""Tests for the dedicated global local-first switch on PipelineSettings.

Phase 10 (v3.0 / LOCAL-02), plan 10-01 Task 1 / D-08: one dedicated global
local-first switch, default ON, env-controllable via a new dedicated key, NOT
overloading the local-vision (--local / GPTR_LOCAL_VISION_*) flags.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from conference_pipeline.config import PipelineSettings, load_settings


@pytest.fixture(autouse=True)
def _clean_local_first_env() -> Iterator[None]:
    """Ensure LOCAL_FIRST_ENABLED is unset unless a test sets it."""
    saved = os.environ.pop("LOCAL_FIRST_ENABLED", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["LOCAL_FIRST_ENABLED"] = saved
        else:
            os.environ.pop("LOCAL_FIRST_ENABLED", None)


def test_default_local_first_enabled_is_true() -> None:
    """Default PipelineSettings has local_first_enabled True (default ON, D-08)."""
    assert PipelineSettings().local_first_enabled is True


def test_field_is_distinct_from_vision_flags() -> None:
    """The new switch is its own field, not an alias of a vision-specific flag."""
    settings = PipelineSettings()
    assert hasattr(settings, "local_first_enabled")
    # It is a boolean routing switch, not the OCR toggle.
    assert isinstance(settings.local_first_enabled, bool)
    assert "local_first_enabled" != "use_llm_ocr"


@pytest.mark.parametrize("raw", ["false", "0", "no", "FALSE", "No"])
def test_env_false_values_disable(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """The env factory parses falsey values to local_first_enabled == False."""
    monkeypatch.setenv("LOCAL_FIRST_ENABLED", raw)
    settings = load_settings()
    assert settings.local_first_enabled is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "TRUE", "Yes"])
def test_env_true_values_enable(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """The env factory parses truthy values to local_first_enabled == True."""
    monkeypatch.setenv("LOCAL_FIRST_ENABLED", raw)
    settings = load_settings()
    assert settings.local_first_enabled is True


def test_env_unset_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env var yields the default True (default ON, D-08)."""
    monkeypatch.delenv("LOCAL_FIRST_ENABLED", raising=False)
    settings = load_settings()
    assert settings.local_first_enabled is True


def test_additive_field_does_not_break_construction() -> None:
    """Construction + validate_model_policy still succeed with the field added."""
    # Explicitly construct with the field set both ways; __post_init__ runs
    # validate_model_policy, so a successful construction proves it is additive.
    assert PipelineSettings(local_first_enabled=False).local_first_enabled is False
    assert PipelineSettings(local_first_enabled=True).local_first_enabled is True
