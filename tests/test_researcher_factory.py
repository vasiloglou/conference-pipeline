"""Phase 46 Plan 02 — `build_researcher` factory tests.

Mirrors the `tests/test_embedding_factory.py` shape: dispatch matrix,
unknown-provider ValueError listing valid set verbatim, non-dict
guard. Covers v1.9 requirement LOCAL-RESEARCH-01 (factory dispatch).
"""

from __future__ import annotations

import pytest

from conference_pipeline.research import (
    FakeResearcher,
    LocalLLMResearcher,
    OpenAIResearcher,
    build_researcher,
)
from conference_pipeline.research.researcher import ResearcherError


# ---------------------------------------------------------------------------
# Dispatch matrix
# ---------------------------------------------------------------------------


def test_build_returns_openai_researcher_when_api_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    researcher = build_researcher({"provider": "openai"})
    assert isinstance(researcher, OpenAIResearcher)


def test_build_openai_propagates_model_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    researcher = build_researcher(
        {"provider": "openai", "model": "gpt-test-model"}
    )
    assert researcher.model == "gpt-test-model"


def test_build_openai_propagates_generation_controls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    researcher = build_researcher(
        {
            "provider": "openai",
            "temperature": 0.0,
            "max_tokens": 8192,
        }
    )
    assert researcher.temperature == 0.0
    assert researcher.max_tokens == 8192


def test_build_openai_raises_without_api_key(monkeypatch):
    """OpenAIResearcher refuses to instantiate without OPENAI_API_KEY.

    The factory does NOT add a special check; it propagates the
    constructor's ResearcherError.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ResearcherError, match="missing_api_key"):
        build_researcher({"provider": "openai"})


def test_build_returns_local_researcher_for_provider_local():
    # Need a fake _client to construct without a real local endpoint.
    class _FakeClient:
        pass
    researcher = build_researcher({"provider": "local", "_client": _FakeClient()})
    assert isinstance(researcher, LocalLLMResearcher)
    assert researcher.provider == "local"


def test_build_local_propagates_base_url_and_models():
    class _FakeClient:
        pass
    researcher = build_researcher(
        {
            "provider": "local",
            "base_url": "http://localhost:7777/v1",
            "fast_model": "qwen-fast",
            "smart_model": "qwen-smart",
            "_client": _FakeClient(),
        }
    )
    assert researcher._base_url == "http://localhost:7777/v1"
    assert researcher.fast_model == "qwen-fast"
    assert researcher.smart_model == "qwen-smart"


def test_build_local_propagates_generation_controls():
    class _FakeClient:
        pass
    researcher = build_researcher(
        {
            "provider": "local",
            "temperature": 0.0,
            "max_tokens": 8192,
            "_client": _FakeClient(),
        }
    )
    assert researcher.temperature == 0.0
    assert researcher.max_tokens == 8192


def test_build_local_explicit_endpoint_label():
    class _FakeClient:
        pass
    researcher = build_researcher(
        {
            "provider": "local",
            "endpoint_label": "test-mlx-endpoint",
            "_client": _FakeClient(),
        }
    )
    assert researcher.endpoint_label == "test-mlx-endpoint"


def test_build_local_rejects_non_loopback_base_url():
    """Factory does not bypass the constructor's loopback validation."""
    with pytest.raises(ResearcherError, match="non_loopback_base_url"):
        build_researcher(
            {
                "provider": "local",
                "base_url": "http://api.openai.com/v1",
            }
        )


def test_build_returns_fake_researcher():
    researcher = build_researcher({"provider": "fake"})
    assert isinstance(researcher, FakeResearcher)


def test_build_fake_accepts_custom_body_template():
    researcher = build_researcher(
        {"provider": "fake", "body_template": "Custom: {topic}"}
    )
    assert researcher.body_template == "Custom: {topic}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_build_unknown_provider_raises_value_error_with_valid_set():
    with pytest.raises(ValueError) as exc_info:
        build_researcher({"provider": "anthropic"})
    msg = str(exc_info.value)
    assert "anthropic" in msg
    assert "'openai', 'local', 'fake'" in msg


def test_build_missing_provider_field_raises_value_error():
    with pytest.raises(ValueError, match=r"requires config\['provider'\]"):
        build_researcher({})


def test_build_non_dict_config_raises_value_error():
    with pytest.raises(ValueError, match="requires a dict"):
        build_researcher("provider=fake")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_local_llm_researcher_exported_from_research_package():
    """v1.9 Phase 46 Plan 02: LocalLLMResearcher + build_researcher in __all__."""
    import conference_pipeline.research as research_pkg
    assert "LocalLLMResearcher" in research_pkg.__all__
    assert "build_researcher" in research_pkg.__all__
    # Importable from the package root.
    from conference_pipeline.research import LocalLLMResearcher, build_researcher
    assert LocalLLMResearcher is research_pkg.LocalLLMResearcher
    assert build_researcher is research_pkg.build_researcher
