"""The production stage table every conference orchestrator dispatches over.

The stage ids, their orders and their receipt filenames are the same for every
conference, and the stage owner registry already declares them
(``stage_bindings.<stage>.stage_order`` / ``receipt_filename``). An
orchestrator therefore does not repeat them: it names the stages it runs, in
dispatch order, together with the remediation text it prints when one blocks
(``next_action``), and :func:`from_registry` fills in the rest.

A conference whose stages are not (yet) bound in the registry supplies a typed
table instead and validates it with :func:`validate_stage_table`.

The table carries no conference literal: everything conference-specific is the
``next_action`` text, which the orchestrator owns.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.production.context import StageSpec


class StageTableError(ValueError):
    """A stage table that disagrees with the registry or with itself."""


@dataclass(frozen=True)
class ProductionStage:
    """A declared production stage and the remediation printed when it blocks.

    Field order is part of the contract: orchestrators construct rows
    positionally (``ProductionStage(id, order, receipt_filename, implemented,
    next_action)``).
    """

    id: str
    order: int
    receipt_filename: str
    implemented: bool
    next_action: str


StageTable = dict[str, ProductionStage]


def registry_stage_bindings(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{stage_id: {"order", "receipt_filename"}}`` for every binding that declares a receipt."""

    rows: dict[str, dict[str, Any]] = {}
    for stage_id, binding in (registry.get("stage_bindings") or {}).items():
        if not isinstance(binding, Mapping):
            continue
        filename = binding.get("receipt_filename")
        order = binding.get("stage_order")
        if not isinstance(filename, str) or not filename:
            continue
        if not isinstance(order, int) or isinstance(order, bool):
            continue
        rows[str(stage_id)] = {"order": order, "receipt_filename": filename}
    return rows


def _load_registry(registry: Mapping[str, Any] | None, registry_path: str | Path | None) -> Mapping[str, Any]:
    if registry is not None:
        return registry
    from conference_pipeline.stage_script_registry import load_stage_script_registry

    return load_stage_script_registry(registry_path)


def from_registry(
    *,
    next_actions: Mapping[str, str],
    not_implemented: Collection[str] = (),
    registry: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
) -> StageTable:
    """Build an orchestrator's stage table from the registry's stage bindings.

    ``next_actions`` names the stages the orchestrator dispatches, in dispatch
    order, each with its remediation text. Every named stage must have a
    registry binding with a ``stage_order`` and a ``receipt_filename``; the
    table fails closed otherwise (an unbound stage would write a receipt that
    no lineage resolver can find). ``not_implemented`` lists declared stages
    the orchestrator does not run yet.
    """

    bindings = registry_stage_bindings(_load_registry(registry, registry_path))
    unknown_not_implemented = sorted(set(not_implemented) - set(next_actions))
    if unknown_not_implemented:
        raise StageTableError(f"not_implemented names undeclared stages: {', '.join(unknown_not_implemented)}")
    table: StageTable = {}
    missing: list[str] = []
    for stage_id, next_action in next_actions.items():
        binding = bindings.get(stage_id)
        if binding is None:
            missing.append(stage_id)
            continue
        if not isinstance(next_action, str) or not next_action.strip():
            raise StageTableError(f"stage {stage_id!r} needs a non-empty next_action")
        table[stage_id] = ProductionStage(
            id=stage_id,
            order=int(binding["order"]),
            receipt_filename=str(binding["receipt_filename"]),
            implemented=stage_id not in not_implemented,
            next_action=next_action,
        )
    if missing:
        raise StageTableError(
            "stage owner registry declares no stage_order/receipt_filename binding for: " + ", ".join(missing)
        )
    return table


def validate_stage_table(
    stages: Mapping[str, StageSpec],
    *,
    registry: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
    require_registry_binding: bool = False,
) -> list[str]:
    """Return every inconsistency in an orchestrator-supplied stage table.

    Keys must equal row ids, receipt filenames must be unique, orders must be
    positive, and a row that the registry also binds must agree with that
    binding. ``require_registry_binding`` additionally rejects rows the
    registry does not bind at all.
    """

    problems: list[str] = []
    seen_filenames: dict[str, str] = {}
    bindings = registry_stage_bindings(_load_registry(registry, registry_path))
    for key, stage in stages.items():
        if key != stage.id:
            problems.append(f"stage table key {key!r} does not match row id {stage.id!r}")
        if not isinstance(stage.order, int) or isinstance(stage.order, bool) or stage.order < 1:
            problems.append(f"stage {stage.id!r} order must be a positive integer")
        previous = seen_filenames.get(stage.receipt_filename)
        if previous is not None:
            problems.append(
                f"stages {previous!r} and {stage.id!r} share receipt filename {stage.receipt_filename!r}"
            )
        seen_filenames[stage.receipt_filename] = stage.id
        binding = bindings.get(stage.id)
        if binding is None:
            if require_registry_binding:
                problems.append(f"stage {stage.id!r} has no registry stage binding")
            continue
        if binding["order"] != stage.order:
            problems.append(f"stage {stage.id!r} order {stage.order} != registry stage_order {binding['order']}")
        if binding["receipt_filename"] != stage.receipt_filename:
            problems.append(
                f"stage {stage.id!r} receipt {stage.receipt_filename!r} != registry "
                f"receipt_filename {binding['receipt_filename']!r}"
            )
    return problems

