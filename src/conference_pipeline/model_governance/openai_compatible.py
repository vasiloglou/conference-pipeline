"""Self-hosted OpenAI-compatible endpoints for ``llm_client``: ``local`` and ``modal`` (v6.1 Phase 4, lane M).

Both speak the same chat-completions protocol (vLLM, llama.cpp, MLX servers), so one adapter
serves both; what differs is where the endpoint is and how it is paid for.

- ``local``: ``GPTR_LOCAL_LLM_BASE_URL`` (default ``http://127.0.0.1:8080/v1``, the repo's
  existing local-researcher setting). Refused unless the URL is loopback: a "local" call that
  reaches another host is a paid or private call wearing the wrong label. Receipts cost $0
  (``local-openai-compatible``).
- ``modal``: ``MODAL_<ALIAS>_BASE_URL`` for the model's catalog alias (``models.yaml``), else
  ``MODAL_BASE_URL``; HTTPS only; the key is the first of ``PROVIDER_CREDENTIALS['modal']``
  that is set. Modal bills GPU time, not tokens: the receipt's cost is the request's wall
  seconds at the catalog's GPU-second price (``pricing_catalog.json``, ``pricing_unit:
  gpu_second``). Wall time includes queueing and cold start, which Modal also bills.

Before the first call to an endpoint, one ``GET <base>/models`` checks that it is up and
learns the served model id (vLLM serves a short name, not the Hugging Face id). A down
endpoint raises ``LLMProviderUnavailable``; the call is never attempted.
"""

from __future__ import annotations

import contextvars
import json
import os
import urllib.request
from typing import Any

from conference_pipeline._security.loopback import is_loopback_url
from conference_pipeline.model_governance.registry import load_model_catalog
from conference_pipeline.provider_receipts import local_openai_chat_completions_create, modal_chat_completions_create

LOCAL_BASE_URL_ENV = "GPTR_LOCAL_LLM_BASE_URL"
LOCAL_DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
MODAL_BASE_URL_ENV = "MODAL_BASE_URL"
PROBE_TIMEOUT_S = {"local": 2.0, "modal": 600.0}  # a cold Modal container takes minutes to answer

_served: dict[str, list[str]] = {}  # base URL -> served model ids, probed once per process
# Per invocation (and per thread): set when this call itself opens a connection to the endpoint, so
# a failed GPU call is charged wall time exactly when it contacted the endpoint (Phase 4 review
# r5 F2, r6 F1, r7 F1-F2).
_contact: contextvars.ContextVar[list[bool] | None] = contextvars.ContextVar("endpoint_contact", default=None)


def _note_contact() -> None:
    box = _contact.get()
    if box is not None:
        box[0] = True


def _lc():
    from conference_pipeline.model_governance import llm_client

    return llm_client


def modal_alias(model: str) -> str | None:
    """The catalog alias of a ``modal`` model, or None."""
    for alias, entry in load_model_catalog().items():
        if entry.get("provider") == "modal" and entry.get("model") == model and entry.get("status") == "active":
            return alias
    return None


def base_url(provider: str, model: str) -> str:
    """The endpoint for this call, or ``LLMConfigError`` naming what is wrong."""
    lc = _lc()
    if provider == "local":
        url = os.environ.get(LOCAL_BASE_URL_ENV) or LOCAL_DEFAULT_BASE_URL
        if not url.startswith(("http://", "https://")) or not is_loopback_url(url):
            raise lc.LLMConfigError(f"{LOCAL_BASE_URL_ENV} must be a loopback http(s) URL for provider 'local'")
        return url.rstrip("/")
    alias = modal_alias(model)
    names = ([f"MODAL_{alias.upper()}_BASE_URL"] if alias else []) + [MODAL_BASE_URL_ENV]
    url = next((os.environ[n] for n in names if os.environ.get(n)), None)
    if not url:
        raise lc.LLMConfigError(f"{' or '.join(names)} is required to call modal model {model!r}")
    if not url.startswith("https://"):
        raise lc.LLMConfigError("a modal endpoint must be https")
    return url.rstrip("/")


def _api_key(provider: str) -> str:
    names = _lc().credential_names(provider) or ()
    return next((os.environ[n] for n in names if os.environ.get(n)), "not-needed")


def probe(provider: str, url: str) -> list[str]:
    """Served model ids at ``url`` (cached); ``LLMProviderUnavailable`` if it does not answer 200."""
    if url in _served:
        return _served[url]
    lc = _lc()
    request = urllib.request.Request(f"{url}/models", headers={"Authorization": f"Bearer {_api_key(provider)}"})
    _note_contact()
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_S[provider]) as response:
            status, body = response.status, response.read()
    except Exception as exc:  # noqa: BLE001 - any transport failure means the endpoint is down
        raise lc.LLMProviderUnavailable(f"{provider} endpoint did not answer: {type(exc).__name__}") from exc
    if status != 200:
        raise lc.LLMProviderUnavailable(f"{provider} endpoint answered HTTP {status}")
    try:
        ids = [str(m["id"]) for m in json.loads(body).get("data", [])]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise lc.LLMProviderUnavailable(f"{provider} endpoint returned no model list") from exc
    _served[url] = ids
    return ids


def served_model(provider: str, model: str, ids: list[str]) -> str:
    """The id to send: the model itself if served, else the endpoint's only model."""
    if model in ids:
        return model
    if len(ids) == 1:
        return ids[0]
    raise _lc().LLMConfigError(f"{provider} endpoint serves {len(ids)} models and not {model!r}")


_transport: Any = None  # tests only: an httpx transport standing in for the network


def _client(provider: str, url: str, request: Any) -> Any:
    from openai import DefaultHttpxClient, OpenAI

    # httpx runs request hooks only when a request is actually being sent, after the SDK has built
    # it, so a request refused locally (serialization, validation) notes no contact (review r8 F1).
    http = DefaultHttpxClient(event_hooks={"request": [lambda _request: _note_contact()]},
                              **({"transport": _transport} if _transport is not None else {}))
    return OpenAI(base_url=url, api_key=_api_key(provider), http_client=http, **_lc()._sdk_options(request))


def _adapter(provider: str, request: Any) -> Any:
    lc = _lc()
    url = base_url(provider, request.model)
    box = [False]
    token = _contact.set(box)
    try:
        return _dispatch(provider, request, url, lc)
    except Exception as exc:
        # A GPU endpoint bills the time of any contact this call made, even when the call fails
        # before a receipt exists or while the receipt is written, so the error says so. A probe
        # answered from the cache contacted nothing.
        # Each adapter states its own contact: a nested call's mark never stands for this one's
        # (review r8 F2); the nested call's producer has already charged it.
        exc.provider_dispatched = box[0]
        raise
    finally:
        _contact.reset(token)


def _dispatch(provider: str, request: Any, url: str, lc: Any) -> Any:
    send_as = served_model(provider, request.model, probe(provider, url))
    kwargs: dict[str, Any] = {
        "model": send_as,
        "messages": lc._chat_messages(request),
        "max_tokens": request.max_tokens,
        "temperature": 0.0 if request.temperature is None else request.temperature,
    }
    if request.thinking is not None:  # Qwen-style chat templates take the switch here
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(request.thinking)}}
    client = _client(provider, url, request)  # its transport hook notes the contact when the request is sent
    if provider == "local":
        response = local_openai_chat_completions_create(client, caller=request.caller, **kwargs)
    else:
        # priced under the catalog model id; the endpoint is sent its served id
        response = modal_chat_completions_create(client, caller=request.caller, _priced_model=request.model, **kwargs)
    return lc._chat_result(response, provider=provider, model=request.model)


def local_adapter(request: Any) -> Any:
    return _adapter("local", request)


def modal_adapter(request: Any) -> Any:
    return _adapter("modal", request)
