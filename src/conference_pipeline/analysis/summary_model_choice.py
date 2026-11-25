"""Which model summarizes the full-read evidence (07A) -- an operator decision, never a default.

Stage 07A (``analysis.exhaustive_topic_summarization``) reads every full-read
evidence unit of a conference once, so its model choice sets both the cost
and the density of everything downstream (07B-07H, the deck blueprints). The
registry lists several allowed selections for it and benchmarks them, but the
trade-off is the operator's, so the stage never picks one:

* ``luna`` -- grounded packs at ~3% of the Gemini Pro cost, fast.
* ``kimi_openrouter`` -- Kimi via OpenRouter: the densest packs (2-3x the
  distinct sources cited, figures quoted), ~25% of the cost, ~5x the latency.
* ``gemini_pro`` -- the legacy default; grounded but thin, and the most
  expensive.
* ``sol`` -- the registered fallback.

(Aliases resolve to concrete models in ``model_governance/models.yaml``.)

The choice comes from an explicit ``--provider`` (optionally ``--model``) on
the command line, or from the conference's ``config.yaml``::

    workflow:
      exhaustive_summary_model: luna   # a catalog alias allowed for the task
      exhaustive_summary_model_decision: "operator, 2026-10-01, ICML topic-09 benchmark"

With neither, the stage refuses before any receipt is touched and asks the
operator. The benchmark behind the options is recorded at
``BENCHMARK_REFERENCE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.model_governance.registry import load_model_catalog, load_registry

TASK_ID = "analysis.exhaustive_topic_summarization"
CONFIG_KEY = "exhaustive_summary_model"
BENCHMARK_REFERENCE = (
    "conferences/icml2026/storage/workflow-runs/icml2026-v4.0-live-production-20260711/"
    "07A-model-benchmark-20261001.json"
)


class SummaryModelNotChosenError(RuntimeError):
    """No 07A model was chosen for this conference; the operator must pick one."""


@dataclass(frozen=True)
class SummaryModelChoice:
    provider: str
    model: str
    alias: str | None
    source: str  # "cli" or "config"
    decision: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "alias": self.alias,
            "source": self.source,
            "decision": self.decision,
        }


def _workflow_config(workspace: Path) -> dict[str, Any]:
    path = Path(workspace) / "config.yaml"
    if not path.is_file():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    workflow = payload.get("workflow") if isinstance(payload, dict) else None
    return workflow if isinstance(workflow, dict) else {}


def allowed_aliases() -> dict[str, tuple[str, str]]:
    """Catalog alias -> (provider, model) for every selection the registry allows for 07A.

    The default, fallback and ``allowed_selections`` of the task, named by the
    catalog alias that resolves to each pair, so an operator records a stable
    alias in config.yaml rather than a model string that a catalog upgrade
    would strand.
    """

    policy = load_registry().get(TASK_ID)
    pairs: list[tuple[str, str]] = [(policy.provider, policy.model)]
    if policy.fallback_provider and policy.fallback_model:
        pairs.append((policy.fallback_provider, policy.fallback_model))
    pairs.extend((item.provider, item.model) for item in policy.allowed_selections)
    by_pair: dict[tuple[str, str], str] = {}
    for alias, entry in load_model_catalog().items():
        if entry.get("status") == "retired":
            continue
        by_pair.setdefault((str(entry["provider"]), str(entry["model"])), alias)
    out: dict[str, tuple[str, str]] = {}
    for pair in pairs:
        alias = by_pair.get(pair)
        if alias is not None:
            out.setdefault(alias, pair)
    return out


def _alias_for(provider: str, model: str) -> str | None:
    for alias, pair in allowed_aliases().items():
        if pair == (provider, model):
            return alias
    return None


def ask_operator_message(workspace: Path) -> str:
    options = ", ".join(
        f"'{alias}' ({provider}/{model})" for alias, (provider, model) in sorted(allowed_aliases().items())
    )
    return (
        f"No 07A summary model is chosen for {Path(workspace).name}. Ask the operator which model "
        f"should summarize the full-read evidence; the registered options are {options}. "
        "Trade-off: luna is grounded and cheapest; kimi_openrouter gives the densest packs "
        "(2-3x the sources cited, figures quoted) at ~25% of the Gemini cost but ~5x the "
        "latency; gemini_pro is the legacy default (grounded, thin, most expensive). "
        f"Record the answer as workflow.{CONFIG_KEY} (plus workflow.{CONFIG_KEY}_decision) in "
        f"{Path(workspace).name}/config.yaml, or pass --provider (and --model). "
        f"Benchmark evidence: {BENCHMARK_REFERENCE}"
    )


def resolve_summary_model(
    workspace: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> SummaryModelChoice:
    """The model for this run: an explicit provider first, then the conference config; else refuse.

    An explicit ``provider`` is the operator's answer on the command line; the
    model then defaults to the task's registered selection for that provider.
    ``model`` without ``provider`` is not a choice (a model string alone does
    not name its route) and is ignored here; the caller validates the pair
    against the registry afterwards either way.
    """

    if provider:
        if model:
            return SummaryModelChoice(provider=provider, model=model, alias=_alias_for(provider, model), source="cli")
        for alias, (allowed_provider, allowed_model) in allowed_aliases().items():
            if allowed_provider == provider:
                return SummaryModelChoice(provider=provider, model=allowed_model, alias=alias, source="cli")
        raise ValueError(f"{TASK_ID} has no registered selection for provider {provider!r}")
    workflow = _workflow_config(workspace)
    configured = workflow.get(CONFIG_KEY)
    if configured in (None, ""):
        raise SummaryModelNotChosenError(ask_operator_message(workspace))
    allowed = allowed_aliases()
    if str(configured) not in allowed:
        raise ValueError(
            f"workflow.{CONFIG_KEY} in config.yaml is {configured!r}; it must be one of "
            f"{sorted(allowed)} (catalog aliases the registry allows for {TASK_ID})"
        )
    chosen_provider, chosen_model = allowed[str(configured)]
    decision = workflow.get(f"{CONFIG_KEY}_decision")
    return SummaryModelChoice(
        provider=chosen_provider,
        model=chosen_model,
        alias=str(configured),
        source="config",
        decision=str(decision) if decision else None,
    )


__all__ = [
    "BENCHMARK_REFERENCE",
    "CONFIG_KEY",
    "SummaryModelChoice",
    "SummaryModelNotChosenError",
    "TASK_ID",
    "allowed_aliases",
    "ask_operator_message",
    "resolve_summary_model",
]
