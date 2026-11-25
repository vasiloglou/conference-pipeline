"""Which research engine a conference uses -- an operator decision, never a default.

Two engines run the deep-dive research (07C), both on the conference's own
hybrid index and the local model:

* ``gpt_researcher`` -- the GPT-Researcher package (planning, sub-queries, deep
  research) with our retriever plugin; gathers more passages, keeps passage
  links, was once derailed by instruction-like text in a retrieved paper.
* ``inhouse`` -- one retrieval plus one local-model synthesis per lane; better
  grounded and quote-heavy in the ICML 2026 topic benchmark (2026-09-30).

Neither is better everywhere, so the pipeline never picks one. The choice comes
from ``--engine`` or from the conference's ``config.yaml``::

    workflow:
      research_engine: inhouse   # or gpt_researcher
      research_engine_decision: "operator, 2026-09-30, ICML topic benchmark"

With neither, the stage refuses before any work and asks the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENGINE_GPT_RESEARCHER = "gpt_researcher"
ENGINE_INHOUSE = "inhouse"
VALID_ENGINES = (ENGINE_GPT_RESEARCHER, ENGINE_INHOUSE)
CONFIG_KEY = "research_engine"
BENCHMARK_REFERENCE = (
    "conferences/icml2026/storage/workflow-runs/icml2026-v4.0-live-production-20260711/"
    "engine-comparison-20260929/"
)


class ResearchEngineNotChosenError(RuntimeError):
    """No research engine was chosen for this conference; the operator must pick one."""


@dataclass(frozen=True)
class EngineChoice:
    engine: str
    source: str  # "cli" or "config"
    decision: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {"engine": self.engine, "source": self.source, "decision": self.decision}


def _workflow_config(workspace: Path) -> dict[str, Any]:
    path = Path(workspace) / "config.yaml"
    if not path.is_file():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    workflow = payload.get("workflow") if isinstance(payload, dict) else None
    return workflow if isinstance(workflow, dict) else {}


def ask_operator_message(workspace: Path) -> str:
    return (
        f"No research engine is chosen for {Path(workspace).name}. Ask the operator which engine "
        f"to use: '{ENGINE_GPT_RESEARCHER}' (the GPT-Researcher package on our index and local "
        f"model: broader retrieval, passage links) or '{ENGINE_INHOUSE}' (one retrieval and one "
        "local-model synthesis per lane: better grounded, quotes its sources). Record the answer as "
        f"workflow.{CONFIG_KEY} in {Path(workspace).name}/config.yaml, or pass --engine. "
        f"Comparison evidence: {BENCHMARK_REFERENCE}"
    )


def resolve_research_engine(workspace: Path, requested: str | None = None) -> EngineChoice:
    """The engine for this run: ``--engine`` first, then the conference config; else refuse."""

    if requested:
        if requested not in VALID_ENGINES:
            raise ValueError(f"engine must be one of {VALID_ENGINES}")
        return EngineChoice(engine=requested, source="cli")
    workflow = _workflow_config(workspace)
    configured = workflow.get(CONFIG_KEY)
    if configured in (None, ""):
        raise ResearchEngineNotChosenError(ask_operator_message(workspace))
    if configured not in VALID_ENGINES:
        raise ValueError(
            f"workflow.{CONFIG_KEY} in config.yaml is {configured!r}; it must be one of {VALID_ENGINES}"
        )
    decision = workflow.get(f"{CONFIG_KEY}_decision")
    return EngineChoice(engine=str(configured), source="config", decision=str(decision) if decision else None)


__all__ = [
    "CONFIG_KEY",
    "ENGINE_GPT_RESEARCHER",
    "ENGINE_INHOUSE",
    "EngineChoice",
    "ResearchEngineNotChosenError",
    "VALID_ENGINES",
    "ask_operator_message",
    "resolve_research_engine",
]
