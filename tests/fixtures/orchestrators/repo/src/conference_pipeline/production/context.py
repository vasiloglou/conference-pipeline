"""Synthetic runner context."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProductionRunnerContext:
    workspace: Path
    run_id: str
    runner_id: str
    receipt_type: str
