"""v6.1 Phase 4 lane P: the generic producer (T-P1..T-P4) and the replay producers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from conference_pipeline.ontology import gold_label
from conference_pipeline.ontology.producer import (
    BudgetExceededError,
    OntologyProducer,
    RefusingProducer,
    SavedProducer,
    outcome_of,
    response_path,
)

TASKS = {"reader": "ontology.read_broad"}


def _fake(calls, text="{}", cost_tokens=(10, 5), stop="end"):
    def complete(task_id, **kwargs):
        calls.append((task_id, kwargs))
        return SimpleNamespace(text=text, stop_reason=stop, provider="p", model="m",
                               input_tokens=cost_tokens[0], output_tokens=cost_tokens[1], receipt=None)
    return complete


def _producer(tmp_path, calls, cap=1.0, cost=0.01, **extra):
    return OntologyProducer(save_root=tmp_path, budget_cap_usd=cap, complete=_fake(calls, **extra.pop("fake", {})),
                            estimate=lambda model, i, o, provider=None: cost, resolve=lambda t: ("p", "m"),
                            tasks=TASKS, caller="ontology.reading", **extra)


def test_t_p1_a_saved_answer_is_reused_by_prompt_and_options(tmp_path):
    calls = []
    p = _producer(tmp_path, calls)
    p("reader", "broad", "W", "prompt one")
    p("reader", "broad", "W", "prompt one")
    assert len(calls) == 1 and calls[0][0] == "ontology.read_broad"
    assert calls[0][1]["caller"] == "ontology.reading.broad"
    other = _producer(tmp_path, calls, role_options={"reader": {"max_tokens": 4000}})
    other("reader", "broad", "W", "prompt one")  # same prompt, other options: not this call's answer
    assert len(calls) == 2


def test_t_p2_an_answer_for_another_prompt_is_kept_as_superseded(tmp_path):
    calls = []
    p = _producer(tmp_path, calls)
    p("reader", "broad", "W", "prompt one")
    p("reader", "broad", "W", "prompt two")
    folder = response_path(tmp_path, "reader", "broad", "W").parent
    assert len(calls) == 2 and len(list(folder.glob("*.superseded-*.json"))) == 1


def test_t_p3_saved_cost_counts_toward_the_cap_on_resume(tmp_path):
    calls = []
    _producer(tmp_path, calls, cost=0.4)("reader", "broad", "W", "a")
    resumed = _producer(tmp_path, calls, cap=0.5, cost=0.4)
    assert resumed.spent_usd == pytest.approx(0.4)
    with pytest.raises(BudgetExceededError):
        resumed("reader", "broad", "W2", "b")
    assert len(calls) == 1


def test_t_p4_the_cap_halts_before_the_call(tmp_path):
    calls = []
    with pytest.raises(BudgetExceededError):
        _producer(tmp_path, calls, cap=0.005)("reader", "broad", "W", "a")
    assert calls == []


def test_outcomes_are_distinct_and_replays_make_no_call(tmp_path):
    assert outcome_of(gold_label.Response("", stop_reason="refusal")) == "refused"
    assert outcome_of(gold_label.Response("")) == "empty"
    assert outcome_of(gold_label.Response("x", stop_reason="max_tokens")) == "truncated"
    assert outcome_of(gold_label.Response("x")) is None
    saved = SavedProducer(tmp_path)
    assert outcome_of(saved("reader", "broad", "W", "p")) == "missing" and saved.calls == 0
    path = response_path(tmp_path, "reader", "broad", "W")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"text": "{\"a\": 1}"}))
    assert saved("reader", "broad", "W", "p").text == "{\"a\": 1}"
    with pytest.raises(AssertionError):
        RefusingProducer()("reader", "broad", "W", "p")


def test_the_labellers_producer_keeps_its_phase_3_defaults(tmp_path):
    live = gold_label.LiveProducer(save_root=tmp_path, budget_cap_usd=1.0)
    assert isinstance(live, OntologyProducer)
    assert live.tasks == gold_label.TASKS and live.role_options["primary"]["reasoning_effort"] == "low"
    assert live.caller == "ontology.gold_label"


def test_review_f1_a_gpu_billed_call_is_reserved_by_wall_time_and_charged_from_its_receipt(tmp_path):
    calls = []

    def complete(task_id, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text="{}", stop_reason="end", provider="modal", model="q", input_tokens=10,
                               output_tokens=5, receipt=SimpleNamespace(estimated_cost_usd=0.013713))

    def producer(cap, root):
        return OntologyProducer(save_root=root, budget_cap_usd=cap, complete=complete,
                                estimate=lambda model, i, o, provider=None: 0.0, resolve=lambda t: ("modal", "q"),
                                gpu_price=lambda provider, model: 0.001097, gpu_wall_cap_s=100.0, tasks=TASKS)

    with pytest.raises(BudgetExceededError):  # (600 s probe + 100 s call) x price > 0: a zero cap refuses
        producer(0.0, tmp_path / "a")("reader", "broad", "W", "p")
    assert calls == []
    ok = producer(1.0, tmp_path / "b")
    response = ok("reader", "broad", "W", "p")
    assert calls[0]["timeout"] == 100.0 and calls[0]["max_retries"] == 0
    assert response.cost_usd == pytest.approx(0.013713) and ok.spent_usd == pytest.approx(0.013713)


def test_review_f1_the_catalog_prices_modal_by_the_gpu_second():
    from conference_pipeline.ontology.producer import gpu_second_price

    assert gpu_second_price("modal", "Qwen/Qwen3.8-27B") == pytest.approx(0.001097)  # model-literal-ok: catalog pin
    assert gpu_second_price("modal", "no-such-model") is None


def _gpu_producer(tmp_path, complete, cap, **extra):
    return OntologyProducer(save_root=tmp_path, budget_cap_usd=cap, complete=complete,
                            estimate=lambda model, i, o, provider=None: 0.0, resolve=lambda t: ("modal", "q"),
                            gpu_price=lambda provider, model: 0.001097, gpu_wall_cap_s=100.0, tasks=TASKS, **extra)


def test_review_r2_f1_a_gpu_call_never_retries_beyond_its_reservation(tmp_path):
    calls = []

    def complete(task_id, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text="{}", stop_reason="end", provider="modal", model="q", input_tokens=1,
                               output_tokens=1, receipt=None)

    _gpu_producer(tmp_path, complete, 1.0, role_options={"reader": {"max_retries": 5}})("reader", "broad", "W", "p")
    assert calls[0]["max_retries"] == 0


def test_review_r2_f2_a_failed_gpu_call_is_charged_from_its_receipt(tmp_path):
    attempts = []

    def complete(task_id, **kwargs):
        attempts.append(1)
        error = RuntimeError("endpoint timed out")
        error.provider_receipt = SimpleNamespace(estimated_cost_usd=0.70)
        raise error

    producer = _gpu_producer(tmp_path, complete, 0.80)
    with pytest.raises(RuntimeError, match="timed out"):
        producer("reader", "broad", "W", "p")
    assert producer.spent_usd == pytest.approx(0.70) and len(producer.receipts) == 1
    with pytest.raises(BudgetExceededError):  # 0.70 spent + a 0.77 reservation crosses 0.80
        producer("reader", "broad", "W", "p")
    assert len(attempts) == 1


def test_review_r3_f1_a_provider_override_is_priced_and_held_to_one_attempt(tmp_path):
    calls = []

    def complete(task_id, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text="{}", stop_reason="end", provider="modal", model="q", input_tokens=1,
                               output_tokens=1, receipt=None)

    def producer(cap, root):  # the default selection is token-priced; the role overrides it to a GPU endpoint
        return OntologyProducer(save_root=root, budget_cap_usd=cap, complete=complete,
                                estimate=lambda model, i, o, provider=None: 0.0, resolve=lambda t: ("anthropic", "c"),
                                gpu_price=lambda provider, model: 0.001097 if provider == "modal" else None,
                                gpu_wall_cap_s=100.0, tasks=TASKS,
                                role_options={"reader": {"provider": "modal", "model": "q", "max_retries": 5}})

    with pytest.raises(BudgetExceededError):  # (600 + 100) s x 0.001097 = 0.77 > 0.20
        producer(0.20, tmp_path / "a")("reader", "broad", "W", "p")
    assert calls == []
    producer(1.0, tmp_path / "b")("reader", "broad", "W", "p")
    assert calls[0]["max_retries"] == 0 and calls[0]["timeout"] == 100.0


def test_review_r3_f2_a_failed_call_charge_survives_a_resume(tmp_path):
    def complete(task_id, **kwargs):
        error = RuntimeError("endpoint timed out")
        error.provider_receipt = SimpleNamespace(estimated_cost_usd=0.70)
        raise error

    with pytest.raises(RuntimeError, match="timed out"):
        _gpu_producer(tmp_path, complete, 0.80)("reader", "broad", "W", "p")
    resumed = _gpu_producer(tmp_path, complete, 0.80)
    assert resumed.spent_usd == pytest.approx(0.70)
    with pytest.raises(BudgetExceededError):
        resumed("reader", "broad", "W", "p")
    assert SavedProducer(tmp_path)("reader", "broad", "W", "p").stop_reason == "missing"  # never read as an answer


def _token_producer(tmp_path, complete, cap, **extra):
    return OntologyProducer(save_root=tmp_path, budget_cap_usd=cap, complete=complete,
                            estimate=lambda model, i, o, provider=None: 0.25, resolve=lambda t: ("p", "m"),
                            gpu_price=lambda provider, model: None, tasks=TASKS, **extra)


def test_review_r4_f1_returning_to_an_earlier_prompt_keeps_every_superseded_charge(tmp_path):
    calls = []
    complete = _fake(calls)
    for prompt in ("a", "b", "a", "b"):
        _token_producer(tmp_path, complete, 1.0)("reader", "broad", "W", prompt)
    assert len(calls) == 4
    resumed = _token_producer(tmp_path, complete, 1.0)
    assert resumed.spent_usd == pytest.approx(1.0)
    with pytest.raises(BudgetExceededError):
        resumed("reader", "broad", "W", "a-fifth-prompt")


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_review_r4_f2_a_gpu_call_is_charged_its_whole_wall_time_probe_included(tmp_path):
    clock = _Clock()

    def complete(task_id, **kwargs):  # 600 s cold-start probe + 90 s generation; the receipt times only 90 s
        clock.now += 690.0
        return SimpleNamespace(text="{}", stop_reason="end", provider="modal", model="q", input_tokens=1,
                               output_tokens=1, receipt=SimpleNamespace(estimated_cost_usd=90 * 0.001097))

    producer = _gpu_producer(tmp_path / "ok", complete, 1.0, clock=clock)
    assert producer("reader", "broad", "W", "p").cost_usd == pytest.approx(690 * 0.001097)

    def probe_times_out(task_id, **kwargs):  # no receipt: the probe raised before the receipted call
        clock.now += 600.0
        error = RuntimeError("modal endpoint did not answer: TimeoutError")
        error.provider_dispatched = True  # as openai_compatible._adapter marks a contacted endpoint
        raise error

    failing = _gpu_producer(tmp_path / "down", probe_times_out, 1.0, clock=clock)
    with pytest.raises(RuntimeError):
        failing("reader", "broad", "W", "p")
    assert failing.spent_usd == pytest.approx(600 * 0.001097)
    assert _gpu_producer(tmp_path / "down", probe_times_out, 1.0).spent_usd == pytest.approx(600 * 0.001097)


def test_review_r4_f3_an_empty_choices_response_carries_its_receipt():
    from conference_pipeline.model_governance import llm_client

    receipt = SimpleNamespace(estimated_cost_usd=0.0987)
    with pytest.raises(IndexError) as caught:
        llm_client._chat_result(SimpleNamespace(choices=[], receipt=receipt), provider="modal", model="q")
    assert caught.value.provider_receipt is receipt


def test_review_r4_f4_an_answer_saved_under_another_task_or_selection_is_not_reused(tmp_path):
    calls = []
    complete = _fake(calls)

    def producer(task, selection):
        return OntologyProducer(save_root=tmp_path, budget_cap_usd=5.0, complete=complete,
                                estimate=lambda model, i, o, provider=None: 0.0, resolve=lambda t: selection,
                                gpu_price=lambda provider, model: None, tasks={"reader": task})

    producer("ontology.read_targeted", ("anthropic", "a"))("reader", "broad", "W", "p")
    producer("ontology.read_targeted", ("anthropic", "a"))("reader", "broad", "W", "p")
    assert len(calls) == 1  # same task and selection: reused
    producer("ontology.read_review", ("anthropic", "a"))("reader", "broad", "W", "p")
    producer("ontology.read_review", ("openai", "o"))("reader", "broad", "W", "p")
    assert [c[0] for c in calls] == ["ontology.read_targeted", "ontology.read_review", "ontology.read_review"]


def test_review_r4_f4_a_response_saved_before_task_and_selection_were_recorded_is_still_reused(tmp_path):
    calls = []
    path = response_path(tmp_path, "reader", "broad", "W")
    path.parent.mkdir(parents=True)
    import hashlib
    path.write_text(json.dumps({"text": "{}", "stop_reason": "end", "cost_usd": 0.1,
                                "prompt_sha256": hashlib.sha256(b"p").hexdigest()}), encoding="utf-8")
    _token_producer(tmp_path, _fake(calls), 1.0)("reader", "broad", "W", "p")
    assert calls == []


def test_review_r5_f1_any_chat_normalization_error_carries_its_receipt():
    from conference_pipeline.model_governance import llm_client

    receipt = SimpleNamespace(estimated_cost_usd=0.0006)
    response = SimpleNamespace(choices=[SimpleNamespace(message=None, finish_reason="stop")], receipt=receipt)
    with pytest.raises(AttributeError) as caught:
        llm_client._chat_result(response, provider="modal", model="q")
    assert caught.value.provider_receipt is receipt


def test_review_r5_f2_a_refusal_before_dispatch_charges_no_gpu_time(tmp_path):
    clock = _Clock()

    def refuses(task_id, **kwargs):  # e.g. missing credentials: no endpoint was contacted
        clock.now += 1.0
        raise RuntimeError("MODAL_API_KEY is required")

    producer = _gpu_producer(tmp_path, refuses, 1.0, clock=clock)
    with pytest.raises(RuntimeError):
        producer("reader", "broad", "W", "p")
    assert producer.spent_usd == 0.0 and not list(tmp_path.rglob("*.failed-*.json"))


def test_review_r5_f2_the_modal_adapter_marks_a_contacted_endpoint(monkeypatch):
    from conference_pipeline.model_governance import openai_compatible as oc

    monkeypatch.setattr(oc, "base_url", lambda provider, model: "https://x.modal.run/v1")

    monkeypatch.delitem(oc._served, "https://x.modal.run/v1", raising=False)

    def endpoint_down(*args, **kwargs):
        raise TimeoutError("endpoint did not answer")

    monkeypatch.setattr(oc.urllib.request, "urlopen", endpoint_down)
    with pytest.raises(Exception) as caught:
        oc.modal_adapter(SimpleNamespace(model="q"))
    assert caught.value.provider_dispatched is True


def test_review_r5_f3_a_refused_replacement_call_keeps_the_reusable_answer(tmp_path):
    calls = []

    def producer(task, cap):
        return OntologyProducer(save_root=tmp_path, budget_cap_usd=cap, complete=_fake(calls),
                                estimate=lambda model, i, o, provider=None: 0.25, resolve=lambda t: ("p", "m"),
                                gpu_price=lambda provider, model: None, tasks={"reader": task})

    producer("task-a", 0.25)("reader", "broad", "W", "p")
    with pytest.raises(BudgetExceededError):
        producer("task-b", 0.25)("reader", "broad", "W", "p")
    producer("task-a", 0.25)("reader", "broad", "W", "p")  # still reused: no second payment
    assert len(calls) == 1 and not list(tmp_path.rglob("*.superseded-*.json"))


def test_review_r5_f4_a_legacy_answer_is_reused_with_no_task_configured(tmp_path):
    import hashlib

    path = response_path(tmp_path, "reader", "broad", "W")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"text": "{}", "stop_reason": "end", "cost_usd": 0.1,
                                "prompt_sha256": hashlib.sha256(b"p").hexdigest()}), encoding="utf-8")
    assert OntologyProducer(save_root=tmp_path, budget_cap_usd=0)("reader", "broad", "W", "p").text == "{}"


def test_review_r6_f1_a_cached_probe_marks_no_dispatch(monkeypatch):
    from conference_pipeline.model_governance import openai_compatible as oc

    url = "https://cached.modal.run/v1"
    monkeypatch.setattr(oc, "base_url", lambda provider, model: url)
    monkeypatch.setitem(oc._served, url, ["other-a", "other-b"])  # cached: no endpoint contact this call

    def no_network(*args, **kwargs):
        raise AssertionError("contacted the endpoint")

    monkeypatch.setattr(oc.urllib.request, "urlopen", no_network)
    with pytest.raises(Exception) as caught:  # served_model refuses: two served ids, neither requested
        oc.modal_adapter(SimpleNamespace(model="q"))
    assert not getattr(caught.value, "provider_dispatched", False)


def _modal_request():
    return SimpleNamespace(model="q", turns=[{"role": "user", "content": "p"}], images=(), system=None,
                           max_tokens=10, temperature=0, thinking=None, caller="t", timeout=None, max_retries=0)


def _chat_transport(sent):
    import httpx

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "q",
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": "{}"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    return httpx.MockTransport(handler)


def test_review_r7_f1_a_sent_request_whose_receipt_fails_is_still_marked(monkeypatch):
    from conference_pipeline import provider_receipts
    from conference_pipeline.model_governance import openai_compatible as oc

    url = "https://cached-ok.modal.run/v1"
    sent = []
    monkeypatch.setattr(oc, "base_url", lambda provider, model: url)
    monkeypatch.setitem(oc._served, url, ["q"])  # cached probe: the request itself is the only contact
    monkeypatch.setattr(oc, "_transport", _chat_transport(sent))

    def receipt_write_fails(*args, **kwargs):  # the request went out; writing its receipt raised
        raise OSError("receipt store unavailable")

    monkeypatch.setattr(provider_receipts, "_finalize_receipt", receipt_write_fails)
    with pytest.raises(OSError) as caught:
        oc.modal_adapter(_modal_request())
    assert len(sent) == 1 and caught.value.provider_dispatched is True


def test_review_r8_f1_a_request_refused_before_transport_marks_no_dispatch(monkeypatch):
    from conference_pipeline.model_governance import openai_compatible as oc

    url = "https://cached-nan.modal.run/v1"
    sent = []
    monkeypatch.setattr(oc, "base_url", lambda provider, model: url)
    monkeypatch.setitem(oc._served, url, ["q"])
    monkeypatch.setattr(oc, "_transport", _chat_transport(sent), raising=False)
    request = _modal_request()
    request.temperature = float("nan")  # the SDK refuses to serialize it: nothing is sent
    with pytest.raises(Exception) as caught:
        oc.modal_adapter(request)
    assert sent == [] and caught.value.provider_dispatched is False


def test_review_r8_f2_a_nested_call_mark_does_not_stand_for_the_outer_call(monkeypatch):
    from conference_pipeline.model_governance import openai_compatible as oc

    outer_url, inner_url = "https://outer.modal.run/v1", "https://inner.modal.run/v1"
    monkeypatch.setitem(oc._served, outer_url, ["q"])  # the outer call's probe is cached
    monkeypatch.delitem(oc._served, inner_url, raising=False)
    monkeypatch.setattr(oc, "base_url", lambda provider, model: inner_url if model == "inner" else outer_url)

    def endpoint_down(*args, **kwargs):
        raise TimeoutError("inner endpoint did not answer")

    monkeypatch.setattr(oc.urllib.request, "urlopen", endpoint_down)
    seen = {}

    def messages_that_call_inside(request):  # the outer call fails inside its own preparation
        try:
            oc.modal_adapter(SimpleNamespace(model="inner"))
        except Exception as exc:
            seen["inner"] = exc.provider_dispatched
            raise

    monkeypatch.setattr(oc._lc(), "_chat_messages", messages_that_call_inside)
    with pytest.raises(Exception) as caught:
        oc.modal_adapter(_modal_request())
    assert seen["inner"] is True and caught.value.provider_dispatched is False


def test_review_r7_f2_a_cache_filled_by_another_caller_marks_no_dispatch(monkeypatch):
    import threading

    from conference_pipeline.model_governance import openai_compatible as oc

    url = "https://raced.modal.run/v1"
    monkeypatch.setattr(oc, "base_url", lambda provider, model: url)
    monkeypatch.delitem(oc._served, url, raising=False)
    real_dispatch, paused, resume = oc._dispatch, threading.Event(), threading.Event()

    def dispatch_after_another_caller(*args):  # A saw an empty cache; B fills it before A probes
        paused.set()
        resume.wait(5)
        return real_dispatch(*args)

    monkeypatch.setattr(oc, "_dispatch", dispatch_after_another_caller)
    monkeypatch.setattr(oc.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("caller A contacted the endpoint"))
    caught = {}

    def caller_a():
        try:
            oc.modal_adapter(SimpleNamespace(model="q"))
        except Exception as exc:  # noqa: BLE001
            caught["exc"] = exc

    thread = threading.Thread(target=caller_a)
    thread.start()
    assert paused.wait(5)
    oc._served[url] = ["other-a", "other-b"]  # caller B's probe result: neither model is requested
    resume.set()
    thread.join(5)
    assert "exc" in caught and not getattr(caught["exc"], "provider_dispatched", False)
