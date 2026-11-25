"""Conference-neutral production-stage orchestration shared by the runners.

Each conference runner (``scripts/run_<slug>_production.py``) owns its stage
table, argparse surface, environment allowlist, and blocker conventions. The
stages that were first written for one runner and then needed by the next --
QA/synthesis, the overview podcast, release, proof, the per-deck NotebookLM
podcast stage, the presentation deck chain -- live here, parameterised by a
:class:`ProductionRunnerContext` so a runner adopts a stage by building a
context, not by copying the stage.
"""

from conference_pipeline.production.context import ProductionRunnerContext, StageSpec

__all__ = ["ProductionRunnerContext", "StageSpec"]
