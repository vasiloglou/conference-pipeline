"""ICML 2026 production runner: pre-spend upstream lineage gate.

The per-command presentation resume tests moved to
``tests/test_presentation_step_lineage.py`` with the runner's retired local
``presentation_command_*`` wrappers (thin-orchestrators W2)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_icml2026_production.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("icml2026_production_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "icml2026"
    (workspace / "decks").mkdir(parents=True)
    (workspace / "analysis").mkdir()
    return workspace


def test_pre_spend_lineage_gate_blocks_a_stage_over_a_stale_upstream_without_a_sticky_blocker(
    tmp_path: Path,
) -> None:
    """v5.1: Stage 7's own upstream moved, so Stage 8 blocks before spend; the
    blocked receipt names the stale chain and no run-wide blocker is written."""

    import json

    runner = load_runner_module()
    workspace = _workspace(tmp_path)
    run_id = "test-icml-upstream-stale"
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    run_dir.mkdir(parents=True)
    extraction_key = f"storage/workflow-runs/{run_id}/14A-exhaustive-topic-extraction-receipt.json"
    (run_dir / "stage-07-analysis.json").write_text(
        json.dumps(
            {
                "stage_id": "analysis",
                "stage_order": 7,
                "status": "succeeded",
                "output_digest": "7" * 64,
                "lineage": {
                    "upstream_receipts": {
                        extraction_key: {
                            "stage_id": "exhaustive-topic-extraction",
                            "digest_kind": "output_digest",
                            "digest": "0" * 64,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stage = runner.STAGES["qa-gap-fill-synthesis"]
    result = runner.block_if_upstream_not_fresh(
        workspace=workspace, run_id=run_id, stage=stage, mode="live", started_at="2026-09-18T00:00:00Z", checks={}
    )

    assert result is not None and result[0] == 1
    receipt = json.loads((run_dir / "stage-08-qa-gap-fill-synthesis.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["checks"]["blocker_code"] == "upstream_lineage_not_fresh"
    codes = {(item["code"], item.get("upstream_stage_id")) for item in receipt["checks"]["upstream_lineage_blockers"]}
    assert ("stage_contract_upstream_stale", "analysis") in codes, codes
    assert runner.active_blockers(workspace, run_id=run_id) == []
