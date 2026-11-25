"""Test helpers for v5.1 receipt lineage.

Runner and contract tests write predecessor stage receipts by hand. Since
verification fails closed on a stage receipt without ``output_digest`` (and
on a receipt under test without ``lineage``), fixtures stamp what a real run
would have recorded: ``stamp_upstream_stage_receipts`` gives every hand-written
stage receipt in a run directory an ``output_digest`` (over its own content)
and a root lineage, and every declared sub-step receipt a root lineage and a fixture
``substep_outputs`` digest; ``bind_fixture_lineage`` binds real lineage onto the
receipt a test is about to verify, exactly as a runner would on a fresh run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conference_pipeline.receipt_lineage import (
    LINEAGE_SCHEMA_VERSION,
    bind_lineage,
    canonical_json_digest,
    content_digest,
    lineage_digest,
    receipt_filenames_from_registry,
    receipt_path_resolver_from_registry,
    substep_specs,
)
from conference_pipeline.stage_script_registry import load_stage_script_registry


def root_lineage() -> dict[str, Any]:
    lineage: dict[str, Any] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "upstream_receipts": {},
        "input_artifacts": {},
        "code_identity": {"runner": "test-fixture"},
        "output_digest_components": {},
        "backfill": None,
    }
    lineage["lineage_digest"] = lineage_digest(lineage)
    return lineage


def stamp_stage_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Give a hand-written stage receipt the fields a real run records."""

    if not isinstance(payload.get("lineage"), dict):
        payload["lineage"] = root_lineage()
    if not isinstance(payload.get("output_digest"), str) or not payload["output_digest"]:
        payload["output_digest"] = content_digest(payload)
    return payload


def stamp_upstream_stage_receipts(run_dir: Path, *, registry_path: str | Path | None = None) -> list[Path]:
    """Stamp every stage receipt in ``run_dir`` (by registry receipt name or
    ``stage-NN-*.json`` shape) that lacks ``output_digest``/``lineage``."""

    registry = load_stage_script_registry(registry_path)
    names = set(receipt_filenames_from_registry(registry).values())
    substep_names = {
        Path(str(spec.get("receipt") or "")).name for spec in substep_specs(registry).values()
    }
    stamped: list[Path] = []
    if not Path(run_dir).is_dir():
        return stamped
    for path in sorted(Path(run_dir).glob("*.json")):
        is_substep = path.name in substep_names
        if path.name not in names and not path.name.startswith("stage-") and not is_substep:
            continue
        if ".superseded-" in path.name or ".attempt-" in path.name or path.name == "summary.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        before = json.dumps(payload, sort_keys=True)
        if is_substep:
            # A hand-written sub-step receipt is a root, as a stage fixture is,
            # with a fixture output digest over its own content.
            if not isinstance(payload.get("lineage"), dict):
                payload["lineage"] = root_lineage()
            if not isinstance(payload.get("substep_outputs"), dict):
                payload["substep_outputs"] = {"artifacts": [], "digest": content_digest(payload)}
        else:
            stamp_stage_receipt(payload)
        if json.dumps(payload, sort_keys=True) != before:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stamped.append(path)
    return stamped


def bind_fixture_lineage(
    receipt: dict[str, Any],
    workspace: Path,
    run_id: str,
    stage_id: str,
    *,
    registry_path: str | Path | None = None,
    profile: str | None = None,
    run_dir: Path | None = None,
    stamp: bool = True,
) -> dict[str, Any]:
    """Bind real lineage (and a fixture output digest) onto a receipt under test.

    ``stamp=False`` leaves the run directory's upstream receipts untouched --
    for fixtures whose upstream is bound by content (the pinned r5 stage-04)
    and whose bytes other checks already pinned.
    """

    directory = run_dir or (Path(workspace) / "storage" / "workflow-runs" / run_id)
    if stamp:
        stamp_upstream_stage_receipts(directory, registry_path=registry_path)
    receipt["lineage"] = bind_lineage(
        workspace,
        run_id,
        stage_id,
        receipt_path_for_stage=receipt_path_resolver_from_registry(workspace, directory, registry_path=registry_path),
        registry_path=registry_path,
        profile=profile,
        code_identity={"runner": "test-fixture"},
    )
    receipt.setdefault("output_digest", canonical_json_digest({"fixture": stage_id}))
    return receipt
