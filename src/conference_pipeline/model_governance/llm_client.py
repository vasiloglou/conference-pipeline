"""Provider-neutral model calls for governed tasks.

A governed task (``registry.yaml``) names which provider/model pairs may do it.
Whether a pair *can* be called is a separate matter, and used to be settled by
whichever SDK a script happened to import: blueprint, figure-spec and
narration stages built an Anthropic client, visual QC built an OpenAI one, so
"any allowed model may do this task" was true on paper and false in code.

``complete`` removes that coupling. One call shape (system, prompt or messages,
optional images, token budget) goes to OpenAI, OpenRouter, Anthropic or Gemini;
every call is receipted through ``provider_receipts`` like the direct calls
were; and the selection is still checked by ``validate_selection``, so a model
the registry has not allowed for the task is refused no matter the transport.

What stays provider-specific is deliberately small and documented per adapter:
how images are attached, how a "refusal" or "truncated" stop is spelled (all
normalised to ``LLMResult.stop_reason``), and which sampling knobs a model
accepts.
"""

from __future__ import annotations

import base64
import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from conference_pipeline.model_governance.model_families import (
    anthropic_rejects_temperature,
    uses_openai_reasoning_params,
)
from conference_pipeline.model_governance.registry import (
    default_model,
    default_model_for_provider,
    default_provider,
    load_registry,
    validate_selection,
)
from conference_pipeline.model_governance.openai_compatible import local_adapter, modal_adapter
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    anthropic_messages_stream_create,
    gemini_generate_content,
    openai_chat_completions_create,
    openrouter_chat_completions_create,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Alternatives: the first variable that is set wins.
PROVIDER_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    # Self-hosted OpenAI-compatible endpoints (openai_compatible.py). A loopback server
    # needs no key; Modal accepts the generic name or the one the v5.2 serving app set.
    "local": (),
    "modal": ("MODAL_API_KEY", "MODAL_QWEN_API_KEY"),
}

Image = tuple[bytes, str]  # (bytes, media type such as "image/png")


class LLMConfigError(RuntimeError):
    """The call cannot be made: no adapter for the provider, or no credential."""


class LLMProviderUnavailable(LLMConfigError):
    """A self-hosted endpoint did not answer its health probe; no call was made."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    # "end" | "max_tokens" | "refusal" | the provider's own word for anything else.
    stop_reason: str
    provider: str
    model: str
    block_types: tuple[str, ...] = ()
    receipt: Any = None
    input_tokens: int = 0
    output_tokens: int = 0


def _token_usage(response: Any) -> tuple[int, int]:
    """(input, output) token counts from any provider's usage block, 0 when absent."""

    def first(source: Any, *names: str) -> int:
        for name in names:
            value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return 0

    usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    from conference_pipeline.provider_receipts import gemini_output_tokens

    output = first(usage, "output_tokens", "completion_tokens")
    if not output:
        output = gemini_output_tokens(usage) or 0  # visible + thinking: both billed
    return first(usage, "input_tokens", "prompt_tokens", "prompt_token_count"), output


def credential_names(provider: str) -> tuple[str, ...] | None:
    return PROVIDER_CREDENTIALS.get(provider)


def has_credentials(provider: str) -> bool:
    names = credential_names(provider)
    if names == ():  # keyless provider (a loopback server)
        return True
    return bool(names) and any(os.environ.get(name) for name in names)


def resolve_selection(
    task_id: str, provider: str | None = None, model: str | None = None
) -> tuple[str, str]:
    """The (provider, model) pair a call for ``task_id`` uses.

    Nothing given: the task's registered default. A model alone is matched
    against the task's own selections, so ``--model openrouter/moonshotai/kimi-k3``
    needs no ``--provider``. A provider alone takes the model the registry
    records for that provider. The result is still checked by
    ``validate_selection`` in :func:`complete`; this only fills in blanks.
    """

    if provider and model:
        return provider, model
    if model:
        policy = load_registry().get(task_id)
        for selection in (policy.default_selection, *policy.allowed_selections):
            if selection.model == model:
                return selection.provider, model
        return default_provider(task_id), model
    if provider:
        return provider, default_model_for_provider(task_id, provider)
    return default_provider(task_id), default_model(task_id)


def _require_credentials(provider: str) -> None:
    names = credential_names(provider)
    if names is None:
        raise LLMConfigError(
            f"no model-call adapter for provider {provider!r}; "
            f"adapters exist for: {', '.join(sorted(PROVIDER_CREDENTIALS))}"
        )
    if not has_credentials(provider):
        raise LLMConfigError(f"{' or '.join(names)} is required to call {provider}")


def _data_uri(image: Image) -> str:
    data, media_type = image
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _normalise_messages(
    prompt: str | None, messages: Sequence[dict[str, Any]] | None
) -> list[dict[str, str]]:
    if (prompt is None) == (messages is None):
        raise ValueError("pass exactly one of prompt= or messages=")
    if messages is None:
        return [{"role": "user", "content": str(prompt)}]
    out = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
    if not out or out[0]["role"] != "user":
        raise ValueError("messages must start with a user turn")
    return out


def complete(
    task_id: str,
    *,
    prompt: str | None = None,
    messages: Sequence[dict[str, Any]] | None = None,
    system: str | None = None,
    images: Sequence[Image] = (),
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 8000,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    thinking: bool | None = None,
    stream: bool = False,
    timeout: float | None = None,
    max_retries: int | None = None,
    caller: str | None = None,
    validate: bool = True,
) -> LLMResult:
    """One governed model call, whichever provider the selection resolves to.

    ``images`` are attached to the first user turn. ``thinking=False`` asks the
    model not to spend its budget on hidden reasoning where the provider lets a
    caller say so (Anthropic's ``thinking: disabled``, or ``effort: low`` for a
    Claude that no longer takes ``disabled``; OpenRouter's
    ``reasoning.enabled=false``); it is ignored elsewhere. ``reasoning_effort``
    is sent to Anthropic as ``output_config.effort``. ``stream`` matters only
    for Anthropic, which wants streaming for very large ``max_tokens``.
    ``temperature`` is not sent to OpenRouter models, whose sampling rules vary
    by upstream (some fix it), nor to Claude generations that reject it.
    """

    provider, model = resolve_selection(task_id, provider, model)
    if validate:
        validate_selection(task_id, provider=provider, model=model)
    _require_credentials(provider)
    turns = _normalise_messages(prompt, messages)
    request = _Request(
        system=system, turns=turns, images=tuple(images), model=model, max_tokens=max_tokens,
        temperature=temperature, reasoning_effort=reasoning_effort, thinking=thinking,
        stream=stream, timeout=timeout, max_retries=max_retries, caller=caller,
    )
    adapter = _ADAPTERS[provider]
    return adapter(request)


@dataclass(frozen=True)
class _Request:
    system: str | None
    turns: list[dict[str, str]]
    images: tuple[Image, ...]
    model: str
    max_tokens: int
    temperature: float | None
    reasoning_effort: str | None
    thinking: bool | None
    stream: bool
    timeout: float | None
    max_retries: int | None
    caller: str | None


def _sdk_options(request: _Request) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if request.timeout is not None:
        options["timeout"] = request.timeout
    if request.max_retries is not None:
        options["max_retries"] = request.max_retries
    return options


# ---- OpenAI-compatible chat completions (OpenAI, OpenRouter) ----------------

def _chat_client(provider: str, request: _Request) -> Any:
    from openai import OpenAI

    options = _sdk_options(request)
    if provider == "openrouter":
        return OpenAI(
            base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], **options
        )
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], **options)


def _chat_messages(request: _Request) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if request.system:
        out.append({"role": "system", "content": request.system})
    for index, turn in enumerate(request.turns):
        if index == 0 and request.images:
            content: Any = [
                *({"type": "image_url", "image_url": {"url": _data_uri(image)}} for image in request.images),
                {"type": "text", "text": turn["content"]},
            ]
        else:
            content = turn["content"]
        out.append({"role": turn["role"], "content": content})
    return out


def openai_completion_options(
    model: str, *, max_completion_tokens: int, temperature: float | None = 0.0
) -> dict[str, object]:
    """Chat Completions options a given OpenAI model accepts."""

    if uses_openai_reasoning_params(model.strip().lower()):
        return {"max_completion_tokens": max_completion_tokens}
    options: dict[str, object] = {"max_tokens": max_completion_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    return options


_CHAT_STOP = {"stop": "end", "length": "max_tokens", "content_filter": "refusal"}


def _receipted(response: Any, build: Any) -> LLMResult:
    """``build()``, with the receipt riding on any error it raises: the call was made and
    billed, so a normalization error is charged like a failed call (Phase 4 review r4 F3,
    r5 F1; v6.1 Phase 5 lane A for the Anthropic and Gemini builders)."""
    try:
        return build()
    except Exception as exc:
        exc.provider_receipt = getattr(response, "receipt", None)
        raise


def _chat_result(response: Any, *, provider: str, model: str) -> LLMResult:
    def build() -> LLMResult:
        choice = response.choices[0]
        raw = str(getattr(choice, "finish_reason", None) or "")
        return LLMResult(
            text=(choice.message.content or "").strip(),
            stop_reason=_CHAT_STOP.get(raw, raw or "end"),
            provider=provider,
            model=model,
            block_types=("text",),
            receipt=getattr(response, "receipt", None),
            input_tokens=_token_usage(response)[0],
            output_tokens=_token_usage(response)[1],
        )
    return _receipted(response, build)


def _openai_adapter(request: _Request) -> LLMResult:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": _chat_messages(request),
        **openai_completion_options(
            request.model,
            max_completion_tokens=request.max_tokens,
            temperature=0.0 if request.temperature is None else request.temperature,
        ),
    }
    if request.reasoning_effort and uses_openai_reasoning_params(request.model.strip().lower()):
        kwargs["reasoning_effort"] = request.reasoning_effort
    response = openai_chat_completions_create(
        _chat_client("openai", request), caller=request.caller, **kwargs
    )
    return _chat_result(response, provider="openai", model=request.model)


def _openrouter_adapter(request: _Request) -> LLMResult:
    """OpenAI-compatible chat completions at OpenRouter (Kimi, GLM, ...).

    Reasoning is opt-in: ``reasoning_effort`` (or the legacy
    ``PRESENTATION_VISUAL_QC_REASONING_EFFORT``-style override a caller passes
    in) sends ``reasoning.effort``; ``thinking=False`` sends
    ``reasoning.enabled=false``. Otherwise the model's own default applies.
    """

    reasoning: dict[str, Any] = {}
    if request.reasoning_effort:
        reasoning["effort"] = request.reasoning_effort
    elif request.thinking is False:
        reasoning["enabled"] = False
    extra = {"extra_body": {"reasoning": reasoning}} if reasoning else {}
    response = openrouter_chat_completions_create(
        _chat_client("openrouter", request),
        caller=request.caller,
        model=request.model,
        messages=_chat_messages(request),
        max_tokens=request.max_tokens,
        **extra,
    )
    return _chat_result(response, provider="openrouter", model=request.model)


# ---- Anthropic ---------------------------------------------------------------

def _anthropic_client(request: _Request) -> Any:
    import anthropic

    return anthropic.Anthropic(**_sdk_options(request))


_ANTHROPIC_STOP = {"end_turn": "end", "stop_sequence": "end", "max_tokens": "max_tokens", "refusal": "refusal"}

# Claude generations differ in how a caller may ask for less reasoning. Older ones
# take ``thinking: disabled``; Opus 5.5 answers that with a 400 and wants
# ``output_config.effort`` instead, and it *thinks by default*, inside the same
# ``max_tokens`` the reply must fit in. The first refusal teaches this process the
# model is adaptive-only, so every later call goes straight to the effort knob.
_ADAPTIVE_THINKING_ONLY: set[str] = set()
_ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _anthropic_effort(value: str | None) -> str | None:
    """A reasoning-effort word as Anthropic spells it, or None when there is none."""

    effort = (value or "").strip().lower()
    if effort in {"minimal", "none"}:
        effort = "low"
    return effort if effort in _ANTHROPIC_EFFORTS else None


def _bad_request_mentions(exc: Exception, *needles: str) -> bool:
    if getattr(exc, "status_code", None) != 400:
        return False
    text = str(exc)
    return any(needle in text for needle in needles)


def _anthropic_send(send: Any, client: Any, request: _Request, kwargs: dict[str, Any]) -> Any:
    """One Anthropic request, retried once when the model refuses a reasoning knob.

    A refused request is a 400 before any generation, so the retry costs nothing
    beyond the round trip.
    """

    try:
        return send(client, caller=request.caller, **kwargs)
    except Exception as exc:  # noqa: BLE001 - classified below; anything else is re-raised
        if kwargs.get("thinking") == {"type": "disabled"} and _bad_request_mentions(exc, "thinking.type.disabled"):
            _ADAPTIVE_THINKING_ONLY.add(request.model)
            kwargs.pop("thinking")
            kwargs.setdefault("output_config", {"effort": "low"})
            return send(client, caller=request.caller, **kwargs)
        if "output_config" in kwargs and _bad_request_mentions(exc, "output_config", "effort"):
            kwargs.pop("output_config")
            return send(client, caller=request.caller, **kwargs)
        raise


def _anthropic_adapter(request: _Request) -> LLMResult:
    turns: list[dict[str, Any]] = []
    for index, turn in enumerate(request.turns):
        if index == 0 and request.images:
            content: Any = [
                *(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.standard_b64encode(data).decode("ascii"),
                        },
                    }
                    for data, media_type in request.images
                ),
                {"type": "text", "text": turn["content"]},
            ]
        else:
            content = turn["content"]
        turns.append({"role": turn["role"], "content": content})
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": turns,
    }
    if request.system:
        kwargs["system"] = request.system
    if request.temperature is not None and not anthropic_rejects_temperature(request.model):
        kwargs["temperature"] = request.temperature
    effort = _anthropic_effort(request.reasoning_effort)
    if request.thinking is False:
        if request.model in _ADAPTIVE_THINKING_ONLY:
            effort = effort or "low"
        else:
            kwargs["thinking"] = {"type": "disabled"}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    client = _anthropic_client(request)
    send = anthropic_messages_stream_create if request.stream else anthropic_messages_create
    try:
        response = _anthropic_send(send, client, request, kwargs)
    except TypeError:
        # A model or SDK version that does not take `thinking` at all.
        if "thinking" not in kwargs:
            raise
        kwargs.pop("thinking")
        response = _anthropic_send(send, client, request, kwargs)
    def build() -> LLMResult:
        blocks = list(getattr(response, "content", None) or [])
        text = "".join(
            str(block.get("text") or "") if isinstance(block, dict) else str(getattr(block, "text", "") or "")
            for block in blocks
        )
        raw = str(getattr(response, "stop_reason", None) or "")
        return LLMResult(
            text=text,
            stop_reason=_ANTHROPIC_STOP.get(raw, raw or "end"),
            provider="anthropic",
            model=request.model,
            block_types=tuple(
                str(block.get("type") if isinstance(block, dict) else getattr(block, "type", type(block).__name__))
                for block in blocks
            ),
            receipt=getattr(response, "receipt", None),
            input_tokens=_token_usage(response)[0],
            output_tokens=_token_usage(response)[1],
        )
    return _receipted(response, build)


# ---- Gemini ------------------------------------------------------------------

def _gemini_client(request: _Request) -> Any:
    from google import genai
    from google.genai import types

    if request.timeout is None:
        return genai.Client()
    return genai.Client(http_options=types.HttpOptions(timeout=int(request.timeout * 1000)))


def _gemini_stop(response: Any) -> tuple[str, str]:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    name = str(getattr(reason, "name", reason) or "")
    if "MAX_TOKENS" in name:
        return "max_tokens", name
    if name in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}:
        return "refusal", name
    return "end", name


# ``reasoning_effort`` -> Gemini ``thinking_level``; an unknown or absent effort sends nothing.
GEMINI_THINKING_LEVELS = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH", "max": "HIGH"}


def _gemini_adapter(request: _Request) -> LLMResult:
    from google.genai import types

    contents = []
    for index, turn in enumerate(request.turns):
        parts = []
        if index == 0:
            parts.extend(
                types.Part.from_bytes(data=data, mime_type=media_type) for data, media_type in request.images
            )
        parts.append(types.Part.from_text(text=turn["content"]))
        contents.append(
            types.Content(role="model" if turn["role"] == "assistant" else "user", parts=parts)
        )
    config_kwargs: dict[str, Any] = {"max_output_tokens": request.max_tokens}
    if request.system:
        config_kwargs["system_instruction"] = request.system
    if request.temperature is not None:
        config_kwargs["temperature"] = request.temperature
    level = GEMINI_THINKING_LEVELS.get(str(request.reasoning_effort or "").lower())
    if level is not None:  # Gemini 3 bills thinking as output; a caller may bound it
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
    response = gemini_generate_content(
        _gemini_client(request),
        caller=request.caller,
        model=request.model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
        # The raw kwargs hold the image bytes; the receipt records digests instead.
        receipt_request_kwargs={
            "model": request.model,
            "image_sha256": [hashlib.sha256(data).hexdigest() for data, _ in request.images],
            "turns": [turn["content"] for turn in request.turns],
        },
    )
    def build() -> LLMResult:
        stop, _raw = _gemini_stop(response)
        return LLMResult(
            text=str(getattr(response, "text", "") or "").strip(),
            stop_reason=stop,
            provider="gemini",
            model=request.model,
            block_types=("text",),
            receipt=getattr(response, "receipt", None),
            input_tokens=_token_usage(response)[0],
            output_tokens=_token_usage(response)[1],
        )
    return _receipted(response, build)


_ADAPTERS = {
    "openai": _openai_adapter,
    "openrouter": _openrouter_adapter,
    "anthropic": _anthropic_adapter,
    "gemini": _gemini_adapter,
    "local": local_adapter,
    "modal": modal_adapter,
}
assert set(_ADAPTERS) == set(PROVIDER_CREDENTIALS)
