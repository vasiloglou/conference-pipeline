"""v6.1 Phase 4 lane M: self-hosted OpenAI-compatible endpoints (``local``, ``modal``) in ``llm_client``.

No test reaches a network: the health probe and the SDK client are fakes.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from conference_pipeline.costs.estimator import PricingCatalog
from conference_pipeline.model_governance import llm_client
from conference_pipeline.model_governance import openai_compatible as oc
from conference_pipeline.model_governance.llm_client import LLMConfigError, LLMProviderUnavailable, complete
from conference_pipeline.model_governance.local_routing import _paid_selection_from_policy
from conference_pipeline.model_governance.registry import model_for_alias
from conference_pipeline.provider_receipts import _estimate_cost, provider_receipt_context

MODAL_TASK = "ai4science.candidate_confirmation"  # the one task whose registry entry allows modal
MODAL_MODEL = model_for_alias("qwen38_27b_modal")


class _Resp(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(ids, status=200, calls=None):
    def open_(request, timeout):
        if calls is not None:
            calls.append((request.full_url, timeout))
        return _Resp(json.dumps({"data": [{"id": i} for i in ids]}).encode(), status)
    return open_


def _fake_sdk(monkeypatch, captured):
    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(id="x", choices=[SimpleNamespace(message=SimpleNamespace(content="ok"),
                                                                finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(oc, "_client", lambda provider, url, request: client)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    oc._served.clear()
    for name in ("GPTR_LOCAL_LLM_BASE_URL", "MODAL_BASE_URL", "MODAL_QWEN38_27B_MODAL_BASE_URL",
                 "MODAL_API_KEY", "MODAL_QWEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with provider_receipt_context(workspace_root=tmp_path, caller="tests.lane_m"):
        yield


def test_t_m1_local_refuses_a_non_loopback_url_before_any_request(monkeypatch):
    monkeypatch.setenv("GPTR_LOCAL_LLM_BASE_URL", "http://10.0.0.5:8080/v1")
    monkeypatch.setattr(oc, "probe", lambda *a, **k: pytest.fail("probed a non-loopback URL"))
    with pytest.raises(LLMConfigError, match="loopback"):
        complete(MODAL_TASK, prompt="x", provider="local", model="m", validate=False)


def test_t_m2_endpoint_down_is_provider_unavailable_after_one_short_probe(monkeypatch):
    calls = []

    def refused(request, timeout):
        calls.append(timeout)
        raise ConnectionRefusedError()
    monkeypatch.setattr(oc, "_client", lambda *a: pytest.fail("called a down endpoint"))
    monkeypatch.setattr(oc.urllib.request, "urlopen", refused)
    with pytest.raises(LLMProviderUnavailable):
        oc._adapter("local", SimpleNamespace(model="m"))
    assert calls == [2.0]
    assert issubclass(LLMProviderUnavailable, LLMConfigError)


def test_local_call_sends_the_served_id_and_costs_nothing(monkeypatch):
    captured = []
    _fake_sdk(monkeypatch, captured)
    monkeypatch.setattr(oc.urllib.request, "urlopen", _opener(["qwen-local"]))
    result = complete(MODAL_TASK, prompt="x", provider="local", model="some/hf-id", validate=False, thinking=False)
    assert captured[0]["model"] == "qwen-local"
    assert captured[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert (result.text, result.provider, result.receipt.estimated_cost_usd) == ("ok", "local", 0.0)


def test_modal_needs_an_https_endpoint_and_a_key(monkeypatch):
    with pytest.raises(LLMConfigError, match="MODAL_API_KEY or MODAL_QWEN_API_KEY"):
        complete(MODAL_TASK, prompt="x", provider="modal", model=MODAL_MODEL)
    monkeypatch.setenv("MODAL_QWEN_API_KEY", "k")
    with pytest.raises(LLMConfigError, match="MODAL_QWEN38_27B_MODAL_BASE_URL or MODAL_BASE_URL"):
        complete(MODAL_TASK, prompt="x", provider="modal", model=MODAL_MODEL)
    monkeypatch.setenv("MODAL_BASE_URL", "http://insecure.example/v1")
    with pytest.raises(LLMConfigError, match="https"):
        complete(MODAL_TASK, prompt="x", provider="modal", model=MODAL_MODEL)


def test_t_m3_modal_is_priced_by_gpu_seconds():
    cost, status, *_ , warnings = _estimate_cost(provider="modal", model=MODAL_MODEL, operation="chat.completions",
                                                 usage={"input_tokens": 9, "output_tokens": 9}, wall_seconds=12.5)
    price = PricingCatalog(json.load(open(oc.__file__.replace("model_governance/openai_compatible.py",
                                                              "costs/pricing_catalog.json"))))
    per_second = price.resolve_model(MODAL_MODEL, "modal")[2]["gpu_second_usd"]
    assert status == "estimated" and cost == round(12.5 * per_second, 6)
    assert any("12.500 s" in w for w in warnings)


def test_modal_call_goes_through_the_alias_endpoint_with_a_long_probe(monkeypatch):
    captured, calls = [], []
    _fake_sdk(monkeypatch, captured)
    monkeypatch.setenv("MODAL_QWEN_API_KEY", "k")
    monkeypatch.setenv("MODAL_QWEN38_27B_MODAL_BASE_URL", "https://example.modal.run/v1/")
    monkeypatch.setattr(oc.urllib.request, "urlopen", _opener(["qwen3.8-27b"], calls=calls))
    result = complete(MODAL_TASK, prompt="x", provider="modal", model=MODAL_MODEL)
    complete(MODAL_TASK, prompt="y", provider="modal", model=MODAL_MODEL)
    assert calls == [("https://example.modal.run/v1/models", 600.0)]  # probed once
    assert captured[0]["model"] == "qwen3.8-27b"
    assert result.receipt.provider == "modal" and result.receipt.cost_status == "estimated"


def test_an_endpoint_serving_several_other_models_is_refused():
    with pytest.raises(LLMConfigError, match="serves 2 models"):
        oc.served_model("local", "m", ["a", "b"])


def test_t_m4_modal_is_a_paid_fallback_candidate_and_local_is_not():
    sel = lambda p: SimpleNamespace(provider=p, model="m")  # noqa: E731
    policy = SimpleNamespace(fallback_selection=sel("local"), default_selection=sel("modal"), allowed_selections=())
    assert _paid_selection_from_policy(policy).provider == "modal"
    policy = SimpleNamespace(fallback_selection=sel("modal"), default_selection=sel("local"), allowed_selections=())
    assert _paid_selection_from_policy(policy).provider == "modal"


def test_t_m5_every_credential_entry_has_an_adapter():
    assert set(llm_client._ADAPTERS) == set(llm_client.PROVIDER_CREDENTIALS)
    assert {"local", "modal"} <= set(llm_client._ADAPTERS)
