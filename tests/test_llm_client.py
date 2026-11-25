"""The provider-neutral model client: one call shape, any allowed provider.

No test here reaches a provider. Each adapter is exercised against a fake SDK
client, so what is asserted is the request the adapter builds and the result it
normalises -- the parts a real call cannot show without spending money.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from conference_pipeline.model_governance import ModelGovernanceError, llm_client
from conference_pipeline.model_governance.llm_client import (
    LLMConfigError,
    complete,
    openai_completion_options,
    resolve_selection,
)

TASK = "presentation.visual_qc"  # allows openai, gemini and (since this change) openrouter
K3 = "openrouter/moonshotai/kimi-k3"
PNG = (b"\x89PNG-bytes", "image/png")


def _chat_response(text: str = "hello", finish: str = "stop"):
    return SimpleNamespace(
        id="gen-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _chat_client(captured: list[dict], **response_kwargs):
    def create(**kwargs):
        captured.append(kwargs)
        return _chat_response(**response_kwargs)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(name, "test-key")


def test_selection_resolution_fills_in_whichever_half_is_missing() -> None:
    assert resolve_selection(TASK, None, K3) == ("openrouter", K3)
    assert resolve_selection(TASK, "openrouter", None) == ("openrouter", K3)
    assert resolve_selection(TASK, "openai", "gpt-6-sol") == ("openai", "gpt-6-sol")


def test_the_registry_still_decides_which_models_may_do_a_task() -> None:
    with pytest.raises(ModelGovernanceError):
        complete(TASK, prompt="x", provider="openrouter", model="openrouter/z-ai/glm-5.3")


def test_a_missing_credential_names_the_variable(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY"):
        complete(TASK, prompt="x", model=K3)


def test_openrouter_sends_the_bare_slug_images_and_no_temperature(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(llm_client, "_chat_client", lambda provider, request: _chat_client(captured))
    result = complete(
        TASK, system="be brief", prompt="judge", images=[PNG], model=K3, max_tokens=123, temperature=0.0,
    )
    (call,) = captured
    assert call["model"] == "moonshotai/kimi-k3"
    assert call["max_tokens"] == 123
    assert "temperature" not in call, "OpenRouter models fix their own sampling"
    assert "extra_body" not in call
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    user = call["messages"][1]["content"]
    assert [block["type"] for block in user] == ["image_url", "text"]
    assert (result.provider, result.model, result.text, result.stop_reason) == ("openrouter", K3, "hello", "end")


def test_openrouter_reasoning_is_sent_only_when_asked_for(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(llm_client, "_chat_client", lambda provider, request: _chat_client(captured))
    complete(TASK, prompt="x", model=K3, reasoning_effort="low")
    complete(TASK, prompt="x", model=K3, thinking=False)
    assert captured[0]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert captured[1]["extra_body"] == {"reasoning": {"enabled": False}}


def test_truncation_and_refusal_are_normalised_across_providers(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client, "_chat_client",
        lambda provider, request: _chat_client([], text="", finish="length"),
    )
    assert complete(TASK, prompt="x", model=K3).stop_reason == "max_tokens"
    monkeypatch.setattr(
        llm_client, "_chat_client",
        lambda provider, request: _chat_client([], text="", finish="content_filter"),
    )
    assert complete(TASK, prompt="x", model=K3).stop_reason == "refusal"


def test_multi_turn_messages_keep_their_order_and_images_ride_the_first_turn(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(llm_client, "_chat_client", lambda provider, request: _chat_client(captured))
    complete(
        TASK, model=K3, images=[PNG],
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "fix it"},
        ],
    )
    roles = [m["role"] for m in captured[0]["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert isinstance(captured[0]["messages"][0]["content"], list)
    assert captured[0]["messages"][2]["content"] == "fix it"


def test_prompt_and_messages_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        complete(TASK, model=K3)
    with pytest.raises(ValueError):
        complete(TASK, model=K3, prompt="x", messages=[{"role": "user", "content": "x"}])


def test_openai_options_follow_the_model_family() -> None:
    assert openai_completion_options("gpt-5.4-mini", max_completion_tokens=1200) == {"max_completion_tokens": 1200}
    assert openai_completion_options("gpt-4o", max_completion_tokens=1200) == {"max_tokens": 1200, "temperature": 0.0}


def _anthropic_client(captured: list[dict], *, stop: str = "end_turn", blocks=None):
    blocks = blocks if blocks is not None else [SimpleNamespace(type="text", text="spec")]

    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            id="msg-1", stop_reason=stop, content=blocks,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_anthropic_adapter_builds_image_blocks_and_can_disable_thinking(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(llm_client, "_anthropic_client", lambda request: _anthropic_client(captured))
    result = complete(
        "presentation.figure_spec_authoring", system="sys", prompt="author", images=[PNG],
        thinking=False, temperature=0.0, max_tokens=500,
    )
    (call,) = captured
    assert call["system"] == "sys" and call["max_tokens"] == 500
    assert call["thinking"] == {"type": "disabled"}
    assert [b["type"] for b in call["messages"][0]["content"]] == ["image", "text"]
    assert (result.provider, result.text, result.stop_reason, result.block_types) == (
        "anthropic", "spec", "end", ("text",),
    )


def test_anthropic_refusal_and_empty_thinking_only_replies_are_visible(monkeypatch) -> None:
    captured: list[dict] = []
    thinking_only = [SimpleNamespace(type="thinking", text="")]
    monkeypatch.setattr(
        llm_client, "_anthropic_client",
        lambda request: _anthropic_client(captured, stop="refusal", blocks=thinking_only),
    )
    result = complete("presentation.figure_spec_authoring", prompt="x")
    assert result.stop_reason == "refusal"
    assert result.text == "" and result.block_types == ("thinking",)


def test_anthropic_retries_without_thinking_when_the_sdk_does_not_take_it(monkeypatch) -> None:
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        if "thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'thinking'")
        return SimpleNamespace(
            id="m", stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(
        llm_client, "_anthropic_client",
        lambda request: SimpleNamespace(messages=SimpleNamespace(create=create)),
    )
    assert complete("presentation.figure_spec_authoring", prompt="x", thinking=False).text == "ok"
    assert "thinking" in calls[0] and "thinking" not in calls[1]


class _BadRequestError(Exception):
    status_code = 400


def _scripted_anthropic(calls: list[dict], refuse):
    """A fake Claude client; ``refuse(kwargs)`` returns an error message or None."""

    def create(**kwargs):
        calls.append(kwargs)
        message = refuse(kwargs)
        if message:
            raise _BadRequestError(message)
        return SimpleNamespace(
            id="m", stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_a_claude_that_refuses_thinking_disabled_is_asked_for_low_effort_instead(monkeypatch) -> None:
    # Opus 5.5 answers `thinking: disabled` with a 400 and thinks by default, inside
    # the same max_tokens the reply must fit in.
    calls: list[dict] = []
    monkeypatch.setattr(llm_client, "_ADAPTIVE_THINKING_ONLY", set())

    def refuse(kwargs):
        if kwargs.get("thinking") == {"type": "disabled"}:
            return '"thinking.type.disabled" is not supported for this model'
        return None

    monkeypatch.setattr(llm_client, "_anthropic_client", lambda request: _scripted_anthropic(calls, refuse))
    task = "presentation.engaging_narration"
    assert complete(task, prompt="x", thinking=False).text == "ok"
    assert calls[0]["thinking"] == {"type": "disabled"} and "output_config" not in calls[0]
    assert "thinking" not in calls[1] and calls[1]["output_config"] == {"effort": "low"}
    # The process remembers: the next call does not send the refused knob at all.
    complete(task, prompt="again", thinking=False)
    assert len(calls) == 3
    assert "thinking" not in calls[2] and calls[2]["output_config"] == {"effort": "low"}


def test_reasoning_effort_reaches_claude_as_output_config_effort(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(llm_client, "_anthropic_client", lambda request: _scripted_anthropic(calls, lambda kw: None))
    complete("presentation.engaging_narration", prompt="x", reasoning_effort="Low")
    complete("presentation.engaging_narration", prompt="x", reasoning_effort="minimal")
    complete("presentation.engaging_narration", prompt="x")
    assert calls[0]["output_config"] == {"effort": "low"}
    assert calls[1]["output_config"] == {"effort": "low"}
    assert "output_config" not in calls[2]


def test_a_claude_without_the_effort_knob_is_retried_without_it(monkeypatch) -> None:
    calls: list[dict] = []

    def refuse(kwargs):
        return "output_config.effort is not supported for this model" if "output_config" in kwargs else None

    monkeypatch.setattr(llm_client, "_anthropic_client", lambda request: _scripted_anthropic(calls, refuse))
    assert complete("presentation.engaging_narration", prompt="x", reasoning_effort="low").text == "ok"
    assert "output_config" in calls[0] and "output_config" not in calls[1]


def test_an_unrelated_bad_request_is_not_swallowed(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_client, "_anthropic_client",
        lambda request: _scripted_anthropic(calls, lambda kw: "prompt is too long"),
    )
    with pytest.raises(_BadRequestError):
        complete("presentation.engaging_narration", prompt="x", thinking=False)
    assert len(calls) == 1


def test_gemini_adapter_attaches_images_and_records_digests_not_bytes(monkeypatch) -> None:
    captured: list[dict] = []

    def generate_content(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            text="verdict", usage_metadata=None,
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
        )

    monkeypatch.setattr(
        llm_client, "_gemini_client",
        lambda request: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )
    result = complete(TASK, system="sys", prompt="judge", images=[PNG], provider="gemini")
    (call,) = captured
    assert call["config"].system_instruction == "sys"
    assert len(call["contents"]) == 1 and len(call["contents"][0].parts) == 2
    assert result.stop_reason == "max_tokens" and result.text == "verdict"


def test_token_usage_is_read_from_each_providers_own_shape(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "_chat_client", lambda provider, request: _chat_client([]))
    result = complete(TASK, prompt="x", model=K3)
    assert (result.input_tokens, result.output_tokens) == (10, 5)

    anthropic_captured: list[dict] = []
    monkeypatch.setattr(llm_client, "_anthropic_client", lambda request: _anthropic_client(anthropic_captured))
    result = complete("presentation.figure_spec_authoring", prompt="x")
    assert (result.input_tokens, result.output_tokens) == (10, 5)

    assert llm_client._token_usage(SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=3))) == (7, 3)
    assert llm_client._token_usage(SimpleNamespace(usage=None)) == (0, 0)


# v6.1 Phase 3 canary (2026-11-18): Gemini bills thinking tokens as output
def test_gemini_thinking_tokens_count_as_output() -> None:
    usage = SimpleNamespace(prompt_token_count=5481, candidates_token_count=317, thoughts_token_count=7683)
    assert llm_client._token_usage(SimpleNamespace(usage_metadata=usage)) == (5481, 8000)
    no_thoughts = SimpleNamespace(prompt_token_count=7, candidates_token_count=3)
    assert llm_client._token_usage(SimpleNamespace(usage_metadata=no_thoughts)) == (7, 3)


def test_reasoning_effort_sets_geminis_thinking_level(monkeypatch) -> None:
    captured: list[dict] = []

    def generate_content(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(text="ok", usage_metadata=None,
                               candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))])

    monkeypatch.setattr(llm_client, "_gemini_client",
                        lambda request: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    complete(TASK, prompt="x", provider="gemini", reasoning_effort="low")
    complete(TASK, prompt="x", provider="gemini")
    assert str(captured[0]["config"].thinking_config.thinking_level).endswith("LOW")
    assert captured[1]["config"].thinking_config is None


# v6.1 Phase 5 lane A (007 open Q3): a reply that arrived and then fails to normalize was
# made and billed, so the error carries its receipt for the producer to charge, as the
# OpenAI-compatible builder already does (Phase 4 review r4 F3, r5 F1).
class _BrokenReply:
    receipt = SimpleNamespace(estimated_cost_usd=0.25)

    @property
    def content(self):
        raise RuntimeError("malformed content blocks")

    @property
    def text(self):
        raise ValueError("no text in this candidate")

    candidates = [SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))]
    usage = None
    usage_metadata = None


def test_phase5_a1_an_anthropic_normalization_error_carries_its_receipt(monkeypatch) -> None:
    reply = _BrokenReply()
    monkeypatch.setattr(llm_client, "_anthropic_client",
                        lambda request: SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: reply)))
    with pytest.raises(RuntimeError) as caught:
        complete("presentation.figure_spec_authoring", prompt="x")
    assert getattr(caught.value, "provider_receipt", None) is not None  # the wrapper's own receipt


def test_phase5_a2_a_gemini_normalization_error_carries_its_receipt(monkeypatch) -> None:
    reply = _BrokenReply()
    monkeypatch.setattr(llm_client, "_gemini_client",
                        lambda request: SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: reply)))
    with pytest.raises(ValueError) as caught:
        complete(TASK, prompt="x", provider="gemini")
    assert getattr(caught.value, "provider_receipt", None) is not None  # the wrapper's own receipt
