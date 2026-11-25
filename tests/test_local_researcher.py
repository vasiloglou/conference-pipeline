"""Phase 46 Plan 01 — LocalLLMResearcher unit tests.

Covers v1.9 requirements LOCAL-RESEARCH-01 (Researcher Protocol impl),
LOCAL-RESEARCH-03 (env var contract), partial coverage of
LOCAL-RESEARCH-04 (loopback validation closes the egress concern).

Tests inject a fake OpenAI-compatible client via the ``_client``
constructor arg so the suite never touches a real local LLM endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from conference_pipeline.research.local_researcher import LocalLLMResearcher
from conference_pipeline.research.researcher import ResearcherError


# ---------------------------------------------------------------------------
# Test doubles — minimal OpenAI-SDK-compatible fake
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChatResponse:
    def __init__(
        self,
        content: str = "synthesized report",
        prompt_tokens: int = 10,
        completion_tokens: int = 50,
    ) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeChatCompletions:
    def __init__(self, response: _FakeChatResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeChatResponse:
        self.calls.append(kwargs)
        return self._response


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeOpenAIClient:
    """Minimal OpenAI-SDK-compatible client for testing."""

    def __init__(
        self,
        response: _FakeChatResponse | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        if raise_exc is not None:
            class _RaisingCompletions:
                def create(self, **kwargs):  # noqa: ANN001, ANN201
                    raise raise_exc
            self.chat = _FakeChat(_RaisingCompletions())  # type: ignore[arg-type]
        else:
            response = response or _FakeChatResponse()
            self.chat = _FakeChat(_FakeChatCompletions(response))


# ---------------------------------------------------------------------------
# Loopback validation (LOCAL-RESEARCH-04)
# ---------------------------------------------------------------------------


def test_constructor_validates_loopback_via_127_0_0_1():
    client = LocalLLMResearcher(base_url="http://127.0.0.1:8080/v1")
    assert client._base_url == "http://127.0.0.1:8080/v1"
    assert client.endpoint_label == "http://localhost:8080"


def test_constructor_validates_loopback_via_localhost():
    client = LocalLLMResearcher(base_url="http://localhost:8080/v1")
    assert client._base_url == "http://localhost:8080/v1"


def test_constructor_rejects_non_loopback_url():
    with pytest.raises(ResearcherError, match="non_loopback_base_url"):
        LocalLLMResearcher(base_url="http://api.openai.com/v1")


def test_constructor_rejects_wildcard_bind():
    with pytest.raises(ResearcherError, match="non_loopback_base_url"):
        LocalLLMResearcher(base_url="http://0.0.0.0:8080/v1")


# ---------------------------------------------------------------------------
# Env-var contract (LOCAL-RESEARCH-03)
# ---------------------------------------------------------------------------


def test_default_base_url_when_env_unset(monkeypatch):
    monkeypatch.delenv("GPTR_LOCAL_LLM_BASE_URL", raising=False)
    client = LocalLLMResearcher()
    assert client._base_url == "http://127.0.0.1:8080/v1"


def test_env_base_url_overrides_default(monkeypatch):
    monkeypatch.setenv("GPTR_LOCAL_LLM_BASE_URL", "http://localhost:9999/v1")
    client = LocalLLMResearcher()
    assert client._base_url == "http://localhost:9999/v1"
    assert client.endpoint_label == "http://localhost:9999"


def test_env_models_override_defaults(monkeypatch):
    monkeypatch.setenv("GPTR_LOCAL_LLM_FAST_MODEL", "qwen-fast")
    monkeypatch.setenv("GPTR_LOCAL_LLM_SMART_MODEL", "qwen-smart")
    client = LocalLLMResearcher()
    assert client.fast_model == "qwen-fast"
    assert client.smart_model == "qwen-smart"
    # Synthesis uses smart by default.
    assert client.model == "qwen-smart"


def test_explicit_constructor_args_override_env(monkeypatch):
    monkeypatch.setenv("GPTR_LOCAL_LLM_FAST_MODEL", "ignored")
    client = LocalLLMResearcher(
        base_url="http://localhost:7777/v1",
        fast_model="explicit-fast",
        smart_model="explicit-smart",
    )
    assert client._base_url == "http://localhost:7777/v1"
    assert client.fast_model == "explicit-fast"
    assert client.smart_model == "explicit-smart"


def test_explicit_endpoint_label_overrides_redaction():
    client = LocalLLMResearcher(
        base_url="http://127.0.0.1:11434/v1",
        endpoint_label="ollama-prod",
    )
    assert client.endpoint_label == "ollama-prod"


# ---------------------------------------------------------------------------
# Synthesis (LOCAL-RESEARCH-01)
# ---------------------------------------------------------------------------


def test_run_returns_response_content():
    fake_client = _FakeOpenAIClient(
        response=_FakeChatResponse(content="my-synthesized-report")
    )
    researcher = LocalLLMResearcher(_client=fake_client)
    result = asyncio.run(researcher.run("what is x?"))
    assert result == "my-synthesized-report"


def test_run_calls_chat_completions_with_smart_model():
    fake_completions = _FakeChatCompletions(_FakeChatResponse())
    fake_client = type("C", (), {"chat": type("Ch", (), {"completions": fake_completions})()})()
    researcher = LocalLLMResearcher(
        _client=fake_client, smart_model="qwen-special-smart",
    )
    asyncio.run(researcher.run("hello"))
    assert len(fake_completions.calls) == 1
    call_kwargs = fake_completions.calls[0]
    assert call_kwargs["model"] == "qwen-special-smart"
    # First message is the REFERENCE_QUALITY_PROMPT system message.
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1] == {"role": "user", "content": "hello"}


def test_run_exposes_temperature_and_max_tokens():
    fake_completions = _FakeChatCompletions(_FakeChatResponse())
    fake_client = type("C", (), {"chat": type("Ch", (), {"completions": fake_completions})()})()
    researcher = LocalLLMResearcher(
        _client=fake_client,
        temperature=0.0,
        max_tokens=8192,
    )
    asyncio.run(researcher.run("hello"))
    call_kwargs = fake_completions.calls[0]
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["max_tokens"] == 8192


def test_run_disables_thinking_mode_for_reasoning_capable_local_models():
    """Regression: reasoning-capable local models (e.g. Qwen3.6-OptiQ) spend the
    whole output budget on a hidden reasoning trace and return an empty
    message.content unless chat_template_kwargs.enable_thinking=False is sent.
    Live discovery: this silently manifested as PaperMatchingError
    "artifacts_store_research_empty" -- retrieval succeeded (12 chunks), but
    the local synthesis call returned "", which callers correctly treat as a
    real "no findings" result rather than a config gap."""
    fake_completions = _FakeChatCompletions(_FakeChatResponse())
    fake_client = type("C", (), {"chat": type("Ch", (), {"completions": fake_completions})()})()
    researcher = LocalLLMResearcher(_client=fake_client)
    asyncio.run(researcher.run("hello"))
    call_kwargs = fake_completions.calls[0]
    assert call_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_run_defaults_max_tokens_when_unset():
    """A reasoning model needs real output headroom even with thinking disabled;
    an unset max_tokens must not silently fall back to a server-side default
    small enough to truncate a full findings synthesis."""
    fake_completions = _FakeChatCompletions(_FakeChatResponse())
    fake_client = type("C", (), {"chat": type("Ch", (), {"completions": fake_completions})()})()
    researcher = LocalLLMResearcher(_client=fake_client)
    asyncio.run(researcher.run("hello"))
    call_kwargs = fake_completions.calls[0]
    assert call_kwargs["max_tokens"] == 16_000


def test_run_tracks_usage_counters():
    fake_client = _FakeOpenAIClient(
        response=_FakeChatResponse(prompt_tokens=42, completion_tokens=100)
    )
    researcher = LocalLLMResearcher(_client=fake_client)
    asyncio.run(researcher.run("query 1"))
    asyncio.run(researcher.run("query 2"))
    usage = researcher.usage()
    assert usage["request_count"] == 2
    assert usage["input_tokens"] == 84
    assert usage["output_tokens"] == 200
    # cost_usd is non-zero only if openai_chat_completions_create's
    # receipt populates it; for a fake client, the receipt path is
    # provider-receipts internal and may be 0.0.
    assert "estimated_cost_usd" in usage
    assert "provider_receipt_paths" in usage


def test_run_wraps_exceptions_in_researcher_error():
    fake_client = _FakeOpenAIClient(raise_exc=RuntimeError("network died"))
    researcher = LocalLLMResearcher(_client=fake_client)
    with pytest.raises(ResearcherError):
        asyncio.run(researcher.run("query"))


def test_local_expansion_client_wraps_call_failure_in_researcher_error():
    """GPTR-03 Phase 11: LocalLLMExpansionClient.expand_queries raises ResearcherError
    when the underlying chat.completions.create call fails — the seam the stage relies on
    to classify a per-topic expansion failure."""
    from conference_pipeline.research.local_researcher import LocalLLMExpansionClient

    client = LocalLLMExpansionClient(
        _client=_FakeOpenAIClient(raise_exc=RuntimeError("boom"))
    )
    with pytest.raises(ResearcherError):
        client.expand_queries("Model Architectures", ["What is attention?"])


def test_local_expansion_client_disables_thinking_mode():
    """Same reasoning-model regression as LocalLLMResearcher: without
    enable_thinking=False the expansion call can return empty content and
    silently produce zero query variants instead of a real error."""
    from conference_pipeline.research.local_researcher import LocalLLMExpansionClient

    fake_completions = _FakeChatCompletions(_FakeChatResponse(content="- variant one\n- variant two"))
    fake_client = type("C", (), {"chat": type("Ch", (), {"completions": fake_completions})()})()
    client = LocalLLMExpansionClient(_client=fake_client)
    client.expand_queries("Model Architectures", ["What is attention?"])
    call_kwargs = fake_completions.calls[0]
    assert call_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_local_expansion_client_sanitizes_secret_in_call_failure():
    """GPTR-03 Phase 11: a secret in the underlying expansion-call error must not leak
    through the ResearcherError message (sanitize_for_receipt)."""
    from conference_pipeline.research.local_researcher import LocalLLMExpansionClient

    fake_secret = "sk-LEAK_EXPANSION_KEY_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    client = LocalLLMExpansionClient(
        _client=_FakeOpenAIClient(
            raise_exc=RuntimeError(f"401 Unauthorized auth={fake_secret}")
        )
    )
    with pytest.raises(ResearcherError) as exc_info:
        client.expand_queries("Model Architectures", ["What is attention?"])
    assert fake_secret not in str(exc_info.value)


def test_run_sanitizes_secrets_in_exception_message():
    fake_secret = "sk-LEAK_TEST_KEY_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    fake_client = _FakeOpenAIClient(
        raise_exc=RuntimeError(f"401 Unauthorized auth={fake_secret}")
    )
    researcher = LocalLLMResearcher(_client=fake_client)
    with pytest.raises(ResearcherError) as exc_info:
        asyncio.run(researcher.run("query"))
    msg = str(exc_info.value)
    assert fake_secret not in msg, (
        "secret leaked through unredacted; sanitize_for_receipt missing"
    )


def test_provider_attribute():
    """LocalLLMResearcher exposes provider="local" for downstream factory dispatch."""
    fake_client = _FakeOpenAIClient()
    researcher = LocalLLMResearcher(_client=fake_client)
    assert researcher.provider == "local"


def test_api_key_is_sentinel_local_only():
    """LocalLLMResearcher uses a sentinel api_key — never reads OPENAI_API_KEY."""
    researcher = LocalLLMResearcher(_client=_FakeOpenAIClient())
    assert researcher.api_key == "local-only"


def test_no_dependency_on_openai_api_key_env(monkeypatch):
    """Setting OPENAI_API_KEY does NOT change LocalLLMResearcher behavior."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-cloud-key")
    fake_client = _FakeOpenAIClient()
    researcher = LocalLLMResearcher(_client=fake_client)
    # Sentinel preserved — OPENAI_API_KEY is not consulted.
    assert researcher.api_key == "local-only"


def test_no_dependency_on_tavily_api_key_env(monkeypatch):
    """LocalLLMResearcher must not read TAVILY_API_KEY (LOCAL-RESEARCH-04)."""
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret")
    # Construction succeeds; no Tavily client created.
    researcher = LocalLLMResearcher(_client=_FakeOpenAIClient())
    assert not hasattr(researcher, "tavily")


def test_usage_shape_matches_openai_researcher():
    """usage() exposes the same key set as OpenAIResearcher.usage()."""
    fake_client = _FakeOpenAIClient()
    researcher = LocalLLMResearcher(_client=fake_client)
    usage = researcher.usage()
    assert set(usage.keys()) == {
        "input_tokens",
        "output_tokens",
        "request_count",
        "estimated_cost_usd",
        "provider_receipt_paths",
        "temperature",
        "max_tokens",
    }


# ---------------------------------------------------------------------------
# v1.9 Phase 49 INT-WARN-02: One-hop loopback import
# ---------------------------------------------------------------------------


def test_local_researcher_imports_is_loopback_url_directly():
    """v1.9 Phase 49 INT-WARN-02 regression: ``local_researcher`` imports
    :func:`is_loopback_url` directly (one hop) from
    :mod:`conference_pipeline._security.loopback`, NOT via the historic
    2-hop alias chain through ``indexing.ollama_embeddings._is_loopback``.

    Asserts:
    - The direct one-hop import line is present.
    - The string ``_is_loopback`` does not appear anywhere in the
      module source (alias usage and docstring references both
      eliminated).
    """
    import inspect

    from conference_pipeline.research import local_researcher

    src = inspect.getsource(local_researcher)
    assert (
        "from conference_pipeline._security.loopback import is_loopback_url"
        in src
    ), (
        "Expected the one-hop direct import "
        "'from conference_pipeline._security.loopback import is_loopback_url' "
        "in local_researcher.py; INT-WARN-02 fix regressed."
    )

    assert "_is_loopback" not in src, (
        "Found '_is_loopback' in local_researcher source; the 2-hop "
        "alias chain via indexing.ollama_embeddings must be gone "
        "(usage and docstring). INT-WARN-02 fix regressed."
    )


def test_ollama_embeddings_no_longer_exports_is_loopback_alias():
    """v1.9 Phase 49 INT-WARN-02 regression: the ``_is_loopback`` alias
    re-export in :mod:`conference_pipeline.indexing.ollama_embeddings`
    has been removed cleanly. No DeprecationWarning shim — callers
    were limited (local_researcher + tests) and were updated in the
    same phase per D-06.
    """
    from conference_pipeline.indexing import ollama_embeddings

    assert not hasattr(ollama_embeddings, "_is_loopback"), (
        "ollama_embeddings still exports '_is_loopback'; the alias "
        "removal per D-06 is incomplete."
    )
