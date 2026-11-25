from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline import receipt_lineage as rl
from conference_pipeline.stage_contracts import verify_stage_contract
from conference_pipeline.stage_script_registry import registry_sha256


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _registry(tmp_path: Path) -> Path:
    """Two production stages: ``up`` (root, one output) and ``down`` (binds ``up``)."""

    payload = {
        "schema_version": 2,
        "registry_name": "test",
        "planning_required_task_ids": [],
        "scripts": {},
        "stage_bindings": {
            "up": {
                "approved_script_ids": [],
                "optional_script_ids": [],
                "required_script_ids": [],
                "forbidden_script_ids": [],
                "receipt_filename": "stage-01-up.json",
                "runtime_contract": {
                    "required_receipts": [],
                    "output_artifact_checks": [["index", "checks.index_path", "checks.index_sha256"]],
                },
            },
            "down": {
                "approved_script_ids": [],
                "optional_script_ids": [],
                "required_script_ids": [],
                "forbidden_script_ids": [],
                "receipt_filename": "stage-02-down.json",
                "runtime_contract": {
                    "required_receipts": [{"path": "storage/workflow-runs/<run-id>/stage-01-up.json"}],
                    "output_artifact_checks": [["report", "checks.report_path", "checks.report_sha256"]],
                },
            },
        },
        "task_bindings": {},
        "skill_contracts": {},
        "evidence_rules": {},
        "tasks": {},
    }
    path = tmp_path / "registry.json"
    _write_json(path, payload)
    return path


def _run(tmp_path: Path, *, down_binds_current: bool = True) -> tuple[Path, Path, Path, Path]:
    workspace = tmp_path / "ws"
    run_id = "run-1"
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    index = workspace / "storage" / "index.json"
    _write_json(index, {"chunks": 3})
    index_sha = rl.sha256_file(index)
    registry = _registry(tmp_path)
    up = run_dir / "stage-01-up.json"
    _write_json(
        up,
        {
            "stage_id": "up",
            "stage_order": 1,
            "status": "succeeded",
            "mode": "live",
            "script_registry_sha256": registry_sha256(registry),
            "required_script_ids": [],
            "started_at": "2026-07-01T00:00:00Z",
            "checks": {"index_path": "storage/index.json", "index_sha256": index_sha},
        },
    )
    report = workspace / "analysis" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# report\n", encoding="utf-8")
    down = run_dir / "stage-02-down.json"
    _write_json(
        down,
        {
            "stage_id": "down",
            "stage_order": 2,
            "status": "succeeded",
            "mode": "live",
            "script_registry_sha256": registry_sha256(registry),
            "required_script_ids": [],
            "started_at": "2026-07-02T00:00:00Z",
            # The evidence a real stage records about what it consumed:
            "checks": {
                "up_index_sha256": index_sha if down_binds_current else "0" * 64,
                "report_path": "analysis/report.md",
                "report_sha256": rl.sha256_file(report),
            },
        },
    )
    return workspace, registry, up, down


def test_backfill_is_read_only_without_apply(tmp_path: Path) -> None:
    workspace, registry, up, down = _run(tmp_path)
    before = {path: path.read_bytes() for path in (up, down)}
    report = rl.backfill_run_lineage(workspace, "run-1", registry_path=registry)
    assert report["applied"] is False
    assert report["stages"]["up"]["action"] == "would_stamp"
    assert report["stages"]["down"]["action"] == "would_stamp"
    assert report["stages"]["down"]["unresolved"] == []
    assert {path: path.read_bytes() for path in (up, down)} == before
    assert not list(up.parent.glob("*.superseded-*")) and not list(up.parent.glob("*.prebackfill-*"))


def test_backfill_binds_downstream_by_recorded_evidence_and_passes_verify(tmp_path: Path) -> None:
    workspace, registry, up, down = _run(tmp_path)
    report = rl.backfill_run_lineage(workspace, "run-1", apply=True, operator="tester", registry_path=registry)
    assert report["stages"]["up"]["action"] == "stamp" and report["stages"]["down"]["action"] == "stamp"
    stamped_up = json.loads(up.read_text(encoding="utf-8"))
    stamped_down = json.loads(down.read_text(encoding="utf-8"))
    assert stamped_up["output_digest"] and stamped_up["lineage"]["upstream_receipts"] == {}
    entry = stamped_down["lineage"]["upstream_receipts"]["storage/workflow-runs/run-1/stage-01-up.json"]
    assert entry["digest"] == stamped_up["output_digest"]
    assert entry["backfill_evidence"]["matched_digests"]
    assert stamped_down["lineage"]["backfill"]["method"] == "evidence_based"
    assert report["stages"]["down"]["verdict"] == "pass", report["stages"]["down"]
    # Prior bytes preserved beside the stamped receipt -- as backfill's own
    # copy of the same execution, never as a superseded one.
    assert report["stages"]["down"]["preserved_prior"].endswith(".json")
    assert list(down.parent.glob("stage-02-down.prebackfill-*.json"))
    assert not list(down.parent.glob("stage-02-down.superseded-*.json"))
    # Verify independently, and confirm a later upstream change is caught.
    assert verify_stage_contract(
        workspace=workspace, run_id="run-1", stage_id="down", receipt=stamped_down, registry_path=registry
    ).status == "pass"
    stamped_up["output_digest"] = "f" * 64
    _write_json(up, stamped_up)
    changed = verify_stage_contract(
        workspace=workspace, run_id="run-1", stage_id="down", receipt=stamped_down, registry_path=registry
    )
    assert [blocker["code"] for blocker in changed.blockers] == ["stage_contract_upstream_changed"]


def test_backfill_marks_unresolved_without_evidence_and_never_fabricates(tmp_path: Path) -> None:
    workspace, registry, up, down = _run(tmp_path, down_binds_current=False)
    report = rl.backfill_run_lineage(workspace, "run-1", apply=True, registry_path=registry)
    assert report["stages"]["down"]["unresolved"] == ["storage/workflow-runs/run-1/stage-01-up.json"]
    stamped_down = json.loads(down.read_text(encoding="utf-8"))
    entry = stamped_down["lineage"]["upstream_receipts"]["storage/workflow-runs/run-1/stage-01-up.json"]
    assert entry["digest"] is None and entry["unresolved"] is True
    assert entry["backfill_evidence"] == "no_matching_digest"
    assert report["stages"]["down"]["verdict"] == "blocked"
    assert "stage_contract_upstream_changed" in report["stages"]["down"]["verdict_blockers"]


def test_backfill_reports_a_consumed_superseded_copy(tmp_path: Path) -> None:
    workspace, registry, up, down = _run(tmp_path, down_binds_current=False)
    # The version ``down`` actually consumed was superseded: its digest lives in the copy.
    old = json.loads(up.read_text(encoding="utf-8"))
    old["checks"]["index_sha256"] = "0" * 64
    _write_json(up.with_name("stage-01-up.superseded-20260701T000000Z.json"), old)
    report = rl.backfill_run_lineage(workspace, "run-1", registry_path=registry)
    entry = report["stages"]["down"]
    assert entry["unresolved"] == ["storage/workflow-runs/run-1/stage-01-up.json"]
    dry = rl.backfill_run_lineage(workspace, "run-1", apply=True, registry_path=registry)
    stamped_down = json.loads(down.read_text(encoding="utf-8"))
    record = stamped_down["lineage"]["upstream_receipts"]["storage/workflow-runs/run-1/stage-01-up.json"]
    assert record["backfill_evidence"] == {"consumed_superseded_copy": "stage-01-up.superseded-20260701T000000Z.json"}
    assert dry["stages"]["down"]["verdict"] == "blocked"


def test_backfill_assume_current_records_operator_attestation(tmp_path: Path) -> None:
    workspace, registry, up, down = _run(tmp_path, down_binds_current=False)
    report = rl.backfill_run_lineage(
        workspace, "run-1", apply=True, assume_current=["down"], operator="nikolaos", registry_path=registry
    )
    stamped_down = json.loads(down.read_text(encoding="utf-8"))
    assert report["stages"]["down"]["unresolved"] == []
    assert stamped_down["lineage"]["backfill"] == {
        "method": "operator_attested",
        "operator": "nikolaos",
        "at": stamped_down["lineage"]["backfill"]["at"],
        "unresolved": [],
    }
    assert report["stages"]["down"]["verdict"] == "pass"


def test_backfill_refuses_non_succeeded_and_is_idempotent(tmp_path: Path) -> None:
    workspace, registry, up, down = _run(tmp_path)
    blocked = json.loads(down.read_text(encoding="utf-8"))
    blocked["status"] = "blocked"
    _write_json(down, blocked)
    report = rl.backfill_run_lineage(workspace, "run-1", apply=True, registry_path=registry)
    assert report["stages"]["down"] == {
        "receipt": "storage/workflow-runs/run-1/stage-02-down.json",
        "action": "refused",
        "status": "blocked",
        "stage_contract_status": None,
    }
    assert "lineage" not in json.loads(down.read_text(encoding="utf-8"))
    again = rl.backfill_run_lineage(workspace, "run-1", apply=True, registry_path=registry)
    assert again["stages"]["up"]["action"] == "skip_already_bound"
    assert len(list(up.parent.glob("stage-01-up.prebackfill-*.json"))) == 1
    forced = rl.backfill_run_lineage(workspace, "run-1", apply=True, force=True, registry_path=registry)
    assert forced["stages"]["up"]["action"] == "stamp"


def test_backfill_stage_filter_and_absent_receipts(tmp_path: Path) -> None:
    workspace, registry, up, down = _run(tmp_path)
    down.unlink()
    report = rl.backfill_run_lineage(workspace, "run-1", stages=["down"], registry_path=registry)
    assert list(report["stages"]) == ["down"]
    assert report["stages"]["down"]["action"] == "absent"


# --------------------------------------------------------------------------
# Review fixes (v5.1 plan 01-05)
# --------------------------------------------------------------------------


def _binding(filename: str, *, order: int, outputs: list[list[str]], requires: list[str] = ()) -> dict:
    return {
        "approved_script_ids": [],
        "optional_script_ids": [],
        "required_script_ids": [],
        "forbidden_script_ids": [],
        "receipt_filename": filename,
        "stage_order": order,
        "runtime_contract": {
            "required_receipts": [{"path": f"storage/workflow-runs/<run-id>/{name}"} for name in requires],
            "output_artifact_checks": outputs,
        },
    }


def _chain_registry(tmp_path: Path, bindings: dict) -> Path:
    payload = {
        "schema_version": 2,
        "registry_name": "test",
        "planning_required_task_ids": [],
        "scripts": {},
        "stage_bindings": bindings,
        "task_bindings": {},
        "skill_contracts": {},
        "evidence_rules": {},
        "tasks": {},
    }
    path = tmp_path / "chain-registry.json"
    _write_json(path, payload)
    return path


def _receipt(stage_id: str, order: int, registry: Path, checks: dict) -> dict:
    return {
        "stage_id": stage_id,
        "stage_order": order,
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(registry),
        "required_script_ids": [],
        "started_at": "2026-07-01T00:00:00Z",
        "checks": checks,
    }


def _synthesis_chain(tmp_path: Path, *, down_checks) -> tuple[Path, Path, Path, dict]:
    """``mid`` (stage 8) writes a summary and reads a manifest; ``down`` (stage 9) reads ``mid``."""

    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    registry = _chain_registry(
        tmp_path,
        {
            "mid": _binding(
                "stage-08-mid.json", order=8, outputs=[["summary", "checks.summary_path", "checks.summary_sha256"]]
            ),
            "down": _binding(
                "stage-09-down.json",
                order=9,
                outputs=[["script", "checks.script_path", "checks.script_sha256"]],
                requires=["stage-08-mid.json"],
            ),
        },
    )
    summary = workspace / "analysis" / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("# regenerated summary\n", encoding="utf-8")
    script = workspace / "analysis" / "script.md"
    script.write_text("# script\n", encoding="utf-8")
    manifest_sha = "c" * 64  # the analysis manifest both stages read
    mid = run_dir / "stage-08-mid.json"
    _write_json(
        mid,
        _receipt(
            "mid",
            8,
            registry,
            {
                "manifest_sha256": manifest_sha,
                "summary_path": "analysis/summary.md",
                "summary_sha256": rl.sha256_file(summary),
            },
        ),
    )
    facts = {
        "manifest_sha": manifest_sha,
        "summary_sha": rl.sha256_file(summary),
        "mid_file_sha": rl.sha256_file(mid),
    }
    checks = {"script_path": "analysis/script.md", "script_sha256": rl.sha256_file(script), **down_checks(facts)}
    down = run_dir / "stage-09-down.json"
    _write_json(down, _receipt("down", 9, registry, checks))
    return workspace, registry, down, facts


def test_backfill_a_value_shared_with_the_upstream_is_not_evidence(tmp_path: Path) -> None:
    # down read the same analysis manifest mid read -- that says nothing about
    # which mid output it consumed.
    workspace, registry, down, _ = _synthesis_chain(
        tmp_path, down_checks=lambda facts: {"manifest_sha256": facts["manifest_sha"]}
    )
    report = rl.backfill_run_lineage(workspace, "run-1", registry_path=registry)
    assert report["stages"]["down"]["unresolved"] == ["storage/workflow-runs/run-1/stage-08-mid.json"]


def test_backfill_a_contradicting_stage_claim_blocks_the_binding(tmp_path: Path) -> None:
    # down recorded the (unchanged) summary sha *and* the receipt sha of a mid
    # version that no longer exists: mid was regenerated after down ran.
    workspace, registry, down, _ = _synthesis_chain(
        tmp_path,
        down_checks=lambda facts: {
            "manifest_sha256": facts["manifest_sha"],
            "summary_sha256": facts["summary_sha"],
            "stage8_receipt_sha256": "a" * 64,
        },
    )
    rl.backfill_run_lineage(workspace, "run-1", apply=True, registry_path=registry)
    stamped = json.loads(down.read_text(encoding="utf-8"))
    entry = stamped["lineage"]["upstream_receipts"]["storage/workflow-runs/run-1/stage-08-mid.json"]
    assert entry["digest"] is None and entry["unresolved"] is True
    assert entry["backfill_evidence"] == {"contradicted_by": ["stage8_receipt_sha256"]}


def test_backfill_value_matching_only_a_superseded_version_blocks_the_binding(tmp_path: Path) -> None:
    workspace, registry, down, facts = _synthesis_chain(
        tmp_path, down_checks=lambda facts: {"summary_sha256": "d" * 64}
    )
    mid = down.parent / "stage-08-mid.json"
    older = json.loads(mid.read_text(encoding="utf-8"))
    older["checks"]["summary_sha256"] = "d" * 64
    older["started_at"] = "2026-06-01T00:00:00Z"
    _write_json(mid.with_name("stage-08-mid.superseded-20260601T000000Z.json"), older)
    report = rl.backfill_run_lineage(workspace, "run-1", apply=True, registry_path=registry)
    assert report["stages"]["down"]["unresolved"] == ["storage/workflow-runs/run-1/stage-08-mid.json"]
    stamped = json.loads(down.read_text(encoding="utf-8"))
    entry = stamped["lineage"]["upstream_receipts"]["storage/workflow-runs/run-1/stage-08-mid.json"]
    assert entry["backfill_evidence"] == {"consumed_superseded_copy": "stage-08-mid.superseded-20260601T000000Z.json"}


def test_backfill_upstream_receipt_sha_evidence_survives_apply_and_force(tmp_path: Path) -> None:
    # Evidence is the upstream's receipt file sha: --apply stamps mid first
    # (changing its bytes), yet down must still bind -- and again on --force.
    workspace, registry, down, _ = _synthesis_chain(
        tmp_path, down_checks=lambda facts: {"stage8_receipt_sha256": facts["mid_file_sha"]}
    )
    dry = rl.backfill_run_lineage(workspace, "run-1", registry_path=registry)
    assert dry["stages"]["down"]["unresolved"] == []
    applied = rl.backfill_run_lineage(workspace, "run-1", apply=True, registry_path=registry)
    assert applied["stages"]["down"]["unresolved"] == []
    assert applied["stages"]["down"]["verdict"] == "pass", applied["stages"]["down"]
    forced = rl.backfill_run_lineage(workspace, "run-1", apply=True, force=True, registry_path=registry)
    assert forced["stages"]["down"]["unresolved"] == []
    assert forced["stages"]["down"]["verdict"] == "pass"
    # backfill's own copies are never mistaken for superseded executions
    assert not list(down.parent.glob("*.superseded-*"))


def test_backfill_and_stage_status_order_dependencies_first_on_stage_order_ties(tmp_path: Path) -> None:
    # "a" sorts before "z" by filename, both are order 10, and "a" requires "z".
    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    registry = _chain_registry(
        tmp_path,
        {
            "a": _binding(
                "stage-10-a.json", order=10, outputs=[["a_out", "checks.a_path"]], requires=["stage-10-z.json"]
            ),
            "z": _binding("stage-10-z.json", order=10, outputs=[["z_out", "checks.z_path", "checks.z_sha256"]]),
        },
    )
    z_out = workspace / "z.txt"
    z_out.parent.mkdir(parents=True, exist_ok=True)
    z_out.write_text("z\n", encoding="utf-8")
    (workspace / "a.txt").write_text("a\n", encoding="utf-8")
    _write_json(
        run_dir / "stage-10-z.json",
        _receipt("z", 10, registry, {"z_path": "z.txt", "z_sha256": rl.sha256_file(z_out)}),
    )
    _write_json(
        run_dir / "stage-10-a.json",
        _receipt("a", 10, registry, {"a_path": "a.txt", "z_sha256": rl.sha256_file(z_out)}),
    )
    report = rl.backfill_run_lineage(workspace, "run-1", apply=True, registry_path=registry)
    assert list(report["stages"]) == ["z", "a"]
    assert report["stages"]["a"]["unresolved"] == []
    assert report["stages"]["a"]["verdict"] == "pass", report["stages"]["a"]
    status = rl.stage_lineage_report(workspace, "run-1", layout="production", registry_path=registry)
    assert list(status["stages"]) == ["z", "a"]
    assert status["next_stage_to_run"] is None
