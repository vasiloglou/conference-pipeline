"""Model-call producers for ontology hosts (v6.1 Phase 4, lane P).

A producer answers ``(role, kind, work, prompt) -> Response``. ``SavedProducer`` replays saved
responses and never calls (a missing one is the ``missing`` outcome); ``RefusingProducer`` raises if
called, for replays that must prove zero calls; ``OntologyProducer`` calls
``llm_client.complete`` under a governed task per role, saves every response keyed by the prompt
and the call options (a response saved for another prompt is kept as superseded, its cost still
counted), counts the cost of responses and failed-call charges already saved on resume, refuses a call whose worst case
would cross the run's cap, and halts if a call cost more than its reservation (R4). Outcomes of a
response: ``accepted`` (``outcome_of`` returns None), ``empty``, ``refused``, ``truncated``,
``missing``; ``unparsable`` is the reader's. They are never retried into acceptance (R8).
Generalised from ``gold_label`` (Phase 3), which keeps its labeller defaults as ``LiveProducer``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

REQUEST_OVERHEAD_TOKENS = 2000  # system and message framing a provider adds to a prompt


@dataclass(frozen=True)
class Response:
    text: str
    stop_reason: str = "end"
    provider: str = "saved"
    model: str = "saved"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Producer(Protocol):
    calls: int

    def __call__(self, role: str, kind: str, work: str, prompt: str) -> Response: ...


def response_path(root: Path, role: str, kind: str, work: str) -> Path:
    safe = hashlib.sha256(work.encode()).hexdigest()[:16]
    return root / role / kind / f"{safe}.json"


class SavedProducer:
    """Replays saved responses; a missing one is an ``empty`` outcome; makes no call."""

    def __init__(self, root: Path) -> None:
        self.root, self.calls = root, 0

    def __call__(self, role: str, kind: str, work: str, prompt: str) -> Response:
        path = response_path(self.root, role, kind, work)
        if not path.is_file():
            return Response(text="", stop_reason="missing")
        saved = json.loads(path.read_text(encoding="utf-8"))
        return Response(text=str(saved.get("text") or ""), stop_reason=str(saved.get("stop_reason") or "end"),
                        provider=str(saved.get("provider") or "saved"), model=str(saved.get("model") or "saved"))


GPU_CALL_TIMEOUT_S = 900.0
GPU_PROBE_BOUND_S = 600.0  # openai_compatible.PROBE_TIMEOUT_S["modal"]: a cold container is billed too


def gpu_second_price(provider: str, model: str) -> float | None:
    """The catalog's USD per GPU second for a GPU-billed model; None for token-priced or unpriced ones."""
    from conference_pipeline.costs.estimator import load_pricing_catalog

    try:
        _, _, data = load_pricing_catalog().resolve_model(model, provider)
    except KeyError:
        return None
    return float(data["gpu_second_usd"]) if data.get("pricing_unit") == "gpu_second" else None


class BudgetExceededError(RuntimeError):
    """The run's spend cap would be crossed; the run halts (R4)."""


@dataclass
class OntologyProducer:
    """``llm_client.complete`` per role's governed task; saves every response; halts at the cap."""

    save_root: Path
    budget_cap_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    complete: Any = None  # injectable for tests; default llm_client.complete
    estimate: Any = None  # injectable; default costs.estimator.estimate_text_model_cost
    max_tokens: int = 8000
    receipts: list[Any] = field(default_factory=list)

    resolve: Any = None  # injectable; default the registry's selection for the role's task
    tasks: dict[str, str] = field(default_factory=dict)  # role -> governed task id
    role_options: dict[str, dict[str, Any]] = field(default_factory=dict)  # role -> call options
    caller: str = "ontology.producer"  # receipt caller prefix; the kind is appended
    gpu_price: Any = None  # injectable; (provider, model) -> USD per GPU second, None when token-priced
    gpu_wall_cap_s: float = GPU_CALL_TIMEOUT_S  # a GPU-billed call's wall-time bound (timeout, no retries)
    clock: Any = None  # injectable monotonic clock; default time.monotonic

    def __post_init__(self) -> None:
        # r1 F9: a resumed run counts what its saved responses already cost.
        for path in sorted(self.save_root.rglob("*.json")) if self.save_root.is_dir() else []:
            try:
                self.spent_usd += float(json.loads(path.read_text(encoding="utf-8")).get("cost_usd") or 0.0)
            except (OSError, ValueError, AttributeError):
                continue

    def _selection(self, role: str, options: dict[str, Any]) -> tuple[str, str]:
        """The (provider, model) the call will use, overrides in ``options`` included (review r3 F1)."""
        provider, model = options.get("provider"), options.get("model")
        if self.resolve is not None:
            default_provider, default_model = self.resolve(self.tasks[role])
            return provider or default_provider, model or default_model
        from conference_pipeline.model_governance.llm_client import resolve_selection
        return resolve_selection(self.tasks[role], provider, model)

    def __call__(self, role: str, kind: str, work: str, prompt: str) -> Response:
        path = response_path(self.save_root, role, kind, work)
        # The reuse key binds the prompt and the call options: an answer made with other thinking
        # settings or another output limit is not this call's answer. Default options hash as the
        # prompt alone, so answers saved before options were recorded still match.
        call_options = {"max_tokens": self.max_tokens, **(self.role_options.get(role) or {})}
        prompt_sha = hashlib.sha256(prompt.encode("utf-8") + b"\0" + json.dumps(
            call_options, sort_keys=True).encode()).hexdigest() if call_options != {"max_tokens": 8000} else \
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        legacy = saved is not None and saved.get("prompt_sha256") == prompt_sha and \
            "task" not in saved and "selection" not in saved
        # review r5 F4: a file saved before task and selection were recorded matches on the prompt key
        # alone, and is reused without resolving a selection (no task need be configured to replay it).
        if not legacy:
            options = dict(self.role_options.get(role) or {})
            max_tokens = int(options.pop("max_tokens", self.max_tokens))
            provider, model = self._selection(role, options)
            task = self.tasks[role]
        # review r4 F4: an answer made under another prompt, governed task or selection is stale.
        stale = saved is not None and not legacy and (
            saved.get("prompt_sha256") != prompt_sha or saved.get("task", task) != task
            or list(saved.get("selection") or [provider, model]) != [provider, model])
        if saved is not None and not stale:  # r2 F1: a saved response is reused, never paid for again
            return Response(text=str(saved.get("text") or ""), stop_reason=str(saved.get("stop_reason") or "end"),
                            provider=str(saved.get("provider") or "saved"), model=str(saved.get("model") or "saved"),
                            input_tokens=int(saved.get("input_tokens") or 0),
                            output_tokens=int(saved.get("output_tokens") or 0), cost_usd=0.0)
        if self.complete is None:
            from conference_pipeline.model_governance.llm_client import complete
            self.complete = complete
        if self.estimate is None:
            from conference_pipeline.costs.estimator import estimate_text_model_cost
            self.estimate = estimate_text_model_cost
        # r1 F9 / r2 F2: refuse a call whose worst case could cross the cap. Input is bounded by the
        # prompt's UTF-8 bytes (a byte-level tokenizer never emits more tokens than bytes) plus a fixed
        # request overhead; output by max_tokens. A real cost above the reservation halts the run.
        reserve = float(self.estimate(model, len(prompt.encode("utf-8")) + REQUEST_OVERHEAD_TOKENS, max_tokens,
                                      provider=provider))
        price = (self.gpu_price or gpu_second_price)(provider, model)
        if price:
            # Phase 4 review F1: a GPU-billed endpoint (Modal) has no token price, so the token bound
            # is 0. Its worst case is the wall time: the readiness probe plus the call, which is held
            # to that bound by a timeout with no retries.
            options.setdefault("timeout", self.gpu_wall_cap_s)
            options["max_retries"] = 0  # review r2 F1: one attempt is what the reservation covers
            reserve = max(reserve, price * (GPU_PROBE_BOUND_S + float(options["timeout"])))
        if self.spent_usd + reserve > self.budget_cap_usd:
            raise BudgetExceededError(f"spent {self.spent_usd:.2f} of a {self.budget_cap_usd:.2f} USD cap; "
                                      f"the next call could cost up to {reserve:.2f}")
        if stale:
            # A stale response is superseded only once its replacement call is admitted (review r5 F3):
            # kept, its cost still counted, never reused, and never overwritten (review r4 F1).
            tag = str(saved.get("prompt_sha256") or "unrecorded")[:12]
            archive, n = path.with_name(f"{path.stem}.superseded-{tag}.json"), 0
            while archive.exists():
                n += 1
                archive = path.with_name(f"{path.stem}.superseded-{tag}-{n}.json")
            path.rename(archive)
        clock = self.clock or time.monotonic
        started = clock()

        def wall_cost() -> float:
            # review r4 F2/F3: a GPU endpoint bills wall time, the readiness probe included, which
            # the receipt does not time; a GPU call is charged at least its whole wall time.
            return price * max(0.0, clock() - started) if price else 0.0

        try:
            result = self.complete(task, prompt=prompt, max_tokens=max_tokens, temperature=0,
                                   caller=f"{self.caller}.{kind}", **options)
        except Exception as exc:
            # review r2 F2: a failed call is still billed (GPU time); its receipt rides on the exception
            receipt = getattr(exc, "provider_receipt", None)
            # review r5 F2: wall time is charged only once the endpoint was contacted (probe or call);
            # a refusal before dispatch (configuration, credentials) costs nothing.
            dispatched = receipt is not None or bool(getattr(exc, "provider_dispatched", False))
            charged = max(float(getattr(receipt, "estimated_cost_usd", None) or 0.0),
                          wall_cost() if dispatched else 0.0)
            if receipt is not None or charged > 0:
                self.calls += 1
                self.receipts.append(receipt)
                self.spent_usd += charged
                # review r3 F2: the charge is saved beside the response path so a resume counts it;
                # the file is never the response itself, so it is never reused as an answer.
                path.parent.mkdir(parents=True, exist_ok=True)
                n = len(list(path.parent.glob(f"{path.stem}.failed-*.json")))
                path.with_name(f"{path.stem}.failed-{n:04d}.json").write_text(json.dumps(
                    {"cost_usd": charged, "reserved_usd": reserve, "prompt_sha256": prompt_sha,
                     "provider": provider, "model": model, "error": type(exc).__name__}, sort_keys=True) + "\n",
                    encoding="utf-8")
            raise
        self.calls += 1
        cost = float(self.estimate(result.model, result.input_tokens, result.output_tokens, provider=result.provider))
        receipt = getattr(result, "receipt", None)
        # The receipt's own cost wins when larger: it carries GPU-second billing, which tokens cannot.
        cost = max(cost, float(getattr(receipt, "estimated_cost_usd", None) or 0.0), wall_cost())
        self.spent_usd += cost
        self.receipts.append(getattr(result, "receipt", None))
        response = Response(text=result.text or "", stop_reason=result.stop_reason, provider=result.provider,
                            model=result.model, input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens, cost_usd=cost)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": response.text, "stop_reason": response.stop_reason,
                                    "provider": response.provider, "model": response.model,
                                    "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                                    "cost_usd": cost, "reserved_usd": reserve, "prompt_sha256": prompt_sha,
                                    "task": task, "selection": [provider, model]},
                                   sort_keys=True,
                                   ensure_ascii=False) + "\n", encoding="utf-8")
        if cost > reserve:
            raise BudgetExceededError(f"a call cost {cost:.4f}, which exceeded its reservation {reserve:.4f}; "
                                      "the run halts for review")
        return response


class RefusingProducer:
    """For replays that must make no call: any call is an error."""

    calls = 0

    def __call__(self, role: str, kind: str, work: str, prompt: str) -> Response:
        raise AssertionError(f"a replay called the producer ({role}, {kind}, {work})")


SavedOnlyProducer = SavedProducer  # the spec's name (Phase 4 §3 lane P)


# --------------------------------------------------------------------------- reading a response


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in ``text`` (code fences allowed); None when there is none."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.S)
    for candidate in (stripped, stripped[stripped.find("{"):stripped.rfind("}") + 1] if "{" in stripped else ""):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def outcome_of(response: Response) -> str | None:
    """An outcome that stops the response from being used, or None when it may be parsed."""
    if response.stop_reason == "refusal":
        return "refused"
    if response.stop_reason == "missing":
        return "missing"
    if not response.text.strip():
        return "empty"
    if response.stop_reason == "max_tokens":
        return "truncated"
    return None
