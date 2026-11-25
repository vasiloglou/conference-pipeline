from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline import receipt_lineage as rl
from conference_pipeline.stage_script_registry import registry_sha256


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _registry(path: Path, bindings: dict[str, dict[str, object]]) -> Path:
    full = {}
    for stage_id, extra in bindings.items():
        full[stage_id] = {
            "approved_script_ids": [],
            "optional_script_ids": [],
            "required_script_ids": [],
            "forbidden_script_ids": [],
            **extra,
        }
    _write_json(
        path,
        {
            "schema_version": 2,
            "registry_name": "test",
            "planning_required_task_ids": [],
            "scripts": {},
            "stage_bindings": full,
            "task_bindings": {},
            "skill_contracts": {},
            "evidence_rules": {},
            "tasks": {},
        },
    )
    return path


def test_compute_output_digest_is_order_independent_and_component_sensitive() -> None:
    a = [{"name": "x", "path": "p/x", "sha256": "1"}, {"name": "y", "path": "p/y", "sha256": "2"}]
    b = list(reversed(a))
    assert rl.compute_output_digest(a) == rl.compute_output_digest(b)
    assert rl.compute_output_digest(a) != rl.compute_output_digest(a, components={"decks": {"01": "d"}})
    assert rl.compute_output_digest([{"name": "x", "path": "p/x", "sha256": "1"}]) != rl.compute_output_digest(a)


def test_digest_output_artifacts_hashes_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "a.json").write_text("{}", encoding="utf-8")
    records = rl.digest_output_artifacts(
        tmp_path,
        [
            {"name": "file", "path": "out.txt"},
            {"name": "dir", "path": "dir"},
            {"name": "pinned", "path": "elsewhere.json", "manifest_sha256": "abc"},
            {"name": "gone", "path": "missing.json"},
        ],
    )
    by_name = {record["name"]: record for record in records}
    assert by_name["file"]["sha256"] == rl.sha256_file(tmp_path / "out.txt")
    assert by_name["file"]["kind"] == "file"
    assert by_name["dir"]["sha256"] == rl.directory_sha256(tmp_path / "dir")
    assert by_name["pinned"]["sha256"] == "abc" and by_name["pinned"]["sha256_source"] == "manifest_sha256"
    assert by_name["gone"]["sha256"] is None and by_name["gone"]["missing"] is True

    (tmp_path / "dir" / "b.json").write_text("{}", encoding="utf-8")
    assert rl.directory_sha256(tmp_path / "dir") != by_name["dir"]["sha256"]


def test_artifacts_from_checks_reads_dotted_keys_and_skips_absent_rows() -> None:
    checks = {
        "index_digest_path": "storage/vector-store/run/index_digest.json",
        "index_digest_sha256": "aaa",
        "nested": {"path": "x.json", "sha": "bbb"},
    }
    table = [
        ["index_digest", "index_digest_path", "index_digest_sha256"],
        ["nested", "nested.path", "nested.sha"],
        ["absent", "no_such_key", "no_such_sha"],
        ["no_sha", "nested.path"],
    ]
    records = rl.artifacts_from_checks(checks, table)
    assert [record["name"] for record in records] == ["index_digest", "nested", "no_sha"]
    assert records[0]["sha256"] == "aaa" and records[1]["sha256"] == "bbb"
    assert "sha256" not in records[2]


def test_content_digest_ignores_volatile_keys() -> None:
    base = {"status": "succeeded", "topic_count": 3}
    assert rl.content_digest({**base, "started_at": "a", "receipt_id": "x", "lineage": {"k": 1}}) == rl.content_digest(base)
    assert rl.content_digest({**base, "topic_count": 4}) != rl.content_digest(base)


def test_bind_lineage_records_required_receipts_upstream_stages_and_extras(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    run_id = "run-1"
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    registry = _registry(
        tmp_path / "registry.json",
        {
            "up": {"receipt_filename": "stage-01-up.json"},
            "sub": {},
            "stage": {
                "runtime_contract": {
                    "required_receipts": [{"path": "storage/workflow-runs/<run-id>/sub-receipt.json"}],
                    "required_upstream_stages": ["up"],
                    "lineage": {"input_artifacts": [{"name": "manifest", "path": "analysis/manifest.json"}]},
                }
            },
        },
    )
    _write_json(run_dir / "stage-01-up.json", {"stage_id": "up", "stage_order": 1, "status": "succeeded", "output_digest": "u1"})
    _write_json(run_dir / "sub-receipt.json", {"status": "succeeded", "count": 2})
    _write_json(workspace / "analysis" / "manifest.json", {"m": 1})
    (workspace / "ledger.json").write_text("{}", encoding="utf-8")

    lineage = rl.bind_lineage(
        workspace,
        run_id,
        "stage",
        registry_path=registry,
        receipt_path_for_stage=rl.receipt_path_resolver_from_registry(workspace, run_dir, registry_path=registry),
        extra_input_artifacts=[{"name": "ledger", "path": "ledger.json"}],
        code_identity={"runner": "test"},
    )
    upstream = lineage["upstream_receipts"]
    assert upstream[f"storage/workflow-runs/{run_id}/stage-01-up.json"] == {
        "stage_id": "up",
        "spec_path": None,
        "source": "required_upstream_stages",
        "digest_kind": "output_digest",
        "digest": "u1",
        "status": "succeeded",
        "optional": False,
        "missing": False,
    }
    sub = upstream[f"storage/workflow-runs/{run_id}/sub-receipt.json"]
    assert sub["digest_kind"] == "content_digest" and sub["digest"] == rl.content_digest({"status": "succeeded", "count": 2})
    assert sub["spec_path"] == "storage/workflow-runs/<run-id>/sub-receipt.json"
    assert lineage["input_artifacts"]["manifest"]["sha256"] == rl.sha256_file(workspace / "analysis" / "manifest.json")
    assert lineage["input_artifacts"]["ledger"]["source"] == "stage"
    assert lineage["code_identity"] == {"script_registry_sha256": registry_sha256(registry), "runner": "test"}
    assert lineage["lineage_digest"] == rl.lineage_digest(lineage)
    assert lineage["binding_source"]["registry_stage_id"] == "stage"


def test_bind_lineage_records_missing_upstream_without_raising(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = _registry(
        tmp_path / "registry.json",
        {"stage": {"runtime_contract": {"required_receipts": [{"path": "storage/workflow-runs/<run-id>/gone.json", "optional": True}]}}},
    )
    lineage = rl.bind_lineage(workspace, "run-1", "stage", registry_path=registry)
    entry = lineage["upstream_receipts"]["storage/workflow-runs/run-1/gone.json"]
    assert entry["missing"] is True and entry["digest"] is None and entry["optional"] is True
    # A missing optional upstream that is still missing verifies; one that appears does not.
    assert rl.verify_lineage(workspace, "run-1", lineage, stage_id="stage") == []
    _write_json(workspace / "storage" / "workflow-runs" / "run-1" / "gone.json", {"status": "succeeded"})
    assert [b["code"] for b in rl.verify_lineage(workspace, "run-1", lineage, stage_id="stage")] == [
        "stage_contract_upstream_changed"
    ]


def test_bind_lineage_records_binding_error_instead_of_raising(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    workspace.mkdir(parents=True)
    registry = _registry(
        tmp_path / "registry.json",
        {"stage": {"workspace_bindings": {"conferences/other2026": {}}, "runtime_contract": {}}},
    )
    lineage = rl.bind_lineage(workspace, "run-1", "stage", registry_path=registry)
    assert "workspace stage binding missing" in lineage["binding_source"]["error"]
    assert lineage["upstream_receipts"] == {} and lineage["lineage_digest"]


def test_lineage_digest_excludes_code_identity_and_is_run_independent() -> None:
    entry = {"stage_id": "up", "spec_path": "storage/workflow-runs/<run-id>/stage-01-up.json", "digest": "u1"}
    one = {
        "upstream_receipts": {"storage/workflow-runs/run-a/stage-01-up.json": entry},
        "input_artifacts": {"m": {"sha256": "s"}},
        "code_identity": {"git_commit": "a"},
    }
    two = {
        "upstream_receipts": {"storage/workflow-runs/run-b/stage-01-up.json": entry},
        "input_artifacts": {"m": {"sha256": "s"}},
        "code_identity": {"git_commit": "b"},
    }
    assert rl.lineage_digest(one) == rl.lineage_digest(two)
    three = {**one, "upstream_receipts": {"storage/workflow-runs/run-a/stage-01-up.json": {**entry, "digest": "u2"}}}
    assert rl.lineage_digest(three) != rl.lineage_digest(one)


def test_verify_lineage_detects_transitive_staleness_and_content_mode_stops_recursion(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    root = run_dir / "stage-01-root.json"
    middle = run_dir / "stage-02-middle.json"
    _write_json(root, {"stage_id": "root", "stage_order": 1, "status": "succeeded", "output_digest": "r1", "lineage": {"upstream_receipts": {}}})
    middle_lineage = {
        "upstream_receipts": {
            "storage/workflow-runs/run-1/stage-01-root.json": {"stage_id": "root", "digest_kind": "output_digest", "digest": "r1"}
        }
    }
    _write_json(middle, {"stage_id": "middle", "stage_order": 2, "status": "succeeded", "output_digest": "m1", "lineage": middle_lineage})
    leaf_lineage = {
        "upstream_receipts": {
            "storage/workflow-runs/run-1/stage-02-middle.json": {"stage_id": "middle", "digest_kind": "output_digest", "digest": "m1"}
        }
    }
    assert rl.verify_lineage(workspace, "run-1", leaf_lineage, stage_id="leaf") == []

    _write_json(root, {"stage_id": "root", "stage_order": 1, "status": "succeeded", "output_digest": "r2", "lineage": {"upstream_receipts": {}}})
    blockers = rl.verify_lineage(workspace, "run-1", leaf_lineage, stage_id="leaf")
    assert [b["code"] for b in blockers] == ["stage_contract_upstream_stale"]
    assert blockers[0]["chain"] == ["storage/workflow-runs/run-1/stage-02-middle.json"]
    assert rl.verify_lineage(workspace, "run-1", leaf_lineage, stage_id="leaf", transitive=False) == []

    # A content-bound upstream is compared by content and never recursed into.
    content_lineage = {
        "upstream_receipts": {
            "storage/workflow-runs/run-1/stage-02-middle.json": {
                "stage_id": "middle",
                "digest_kind": "content_digest",
                "digest": rl.content_digest(json.loads(middle.read_text(encoding="utf-8"))),
            }
        }
    }
    assert rl.verify_lineage(workspace, "run-1", content_lineage, stage_id="leaf") == []


def test_verify_lineage_blocks_stage_receipt_without_output_digest(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write_json(
        workspace / "storage" / "workflow-runs" / "run-1" / "stage-01-up.json",
        {"stage_id": "up", "stage_order": 1, "status": "succeeded"},
    )
    lineage = {
        "upstream_receipts": {
            "storage/workflow-runs/run-1/stage-01-up.json": {"stage_id": "up", "digest_kind": "output_digest", "digest": "x"}
        }
    }
    assert [b["code"] for b in rl.verify_lineage(workspace, "run-1", lineage, stage_id="s")] == [
        "stage_contract_upstream_output_digest_missing"
    ]
    assert [b["code"] for b in rl.verify_lineage(workspace, "run-1", None, stage_id="s")] == [
        "stage_contract_lineage_missing"
    ]


def test_preserve_prior_receipt_archives_by_prior_status_with_sequence(tmp_path: Path) -> None:
    canonical = tmp_path / "stage-07-analysis.json"
    assert rl.preserve_prior_receipt(canonical, new_status="succeeded") is None

    _write_json(canonical, {"status": "succeeded", "n": 1})
    first = rl.preserve_prior_receipt(canonical, new_status="succeeded", stamp="20260918T000000Z")
    assert first == tmp_path / "stage-07-analysis.superseded-20260918T000000Z.json"
    assert json.loads(first.read_text(encoding="utf-8")) == {"status": "succeeded", "n": 1}
    second = rl.preserve_prior_receipt(canonical, new_status="succeeded", stamp="20260918T000000Z")
    assert second == tmp_path / "stage-07-analysis.superseded-20260918T000000Z-1.json"

    _write_json(canonical, {"status": "blocked", "n": 2})
    attempt = rl.preserve_prior_receipt(canonical, new_status="succeeded", stamp="20260918T000001Z")
    assert attempt == tmp_path / "stage-07-analysis.attempt-20260918T000001Z.json"
    # The canonical file is untouched; the caller writes the new receipt.
    assert json.loads(canonical.read_text(encoding="utf-8")) == {"status": "blocked", "n": 2}


def test_apply_lineage_sets_output_fields(tmp_path: Path) -> None:
    receipt = {"status": "succeeded"}
    lineage = {"upstream_receipts": {}, "input_artifacts": {}, "output_digest_components": {}}
    rl.apply_lineage(
        receipt,
        lineage=lineage,
        output_artifacts=[{"name": "a", "path": "a.json", "sha256": "1"}],
        components={"decks": {"01": "d1"}},
    )
    assert receipt["output_artifacts"] == [{"name": "a", "path": "a.json", "sha256": "1"}]
    assert receipt["lineage"]["output_digest_components"] == {"decks": {"01": "d1"}}
    assert receipt["output_digest"] == rl.compute_output_digest(
        receipt["output_artifacts"], components={"decks": {"01": "d1"}}
    )
    assert lineage["output_digest_components"] == {}  # caller's mapping not mutated


def test_receipt_filenames_from_real_registry_cover_the_production_stages() -> None:
    from conference_pipeline.stage_script_registry import load_stage_script_registry

    names = rl.receipt_filenames_from_registry(load_stage_script_registry())
    assert names["corpus-inventory-index"] == "stage-06-corpus-inventory-index.json"
    assert names["exhaustive-topic-extraction"] == "14A-exhaustive-topic-extraction-receipt.json"
    assert names["proof"] == "stage-12-proof.json"
    assert len(set(names.values())) == len(names)


def test_layout_scoped_upstream_stage_is_bound_and_enforced_only_in_its_layout(tmp_path: Path) -> None:
    from conference_pipeline.stage_contracts import verify_stage_contract

    workspace = tmp_path / "ws"
    run_id = "run-1"
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    registry = _registry(
        tmp_path / "registry.json",
        {
            "index": {"receipt_filename": "stage-06-index.json"},
            "extract": {
                "runtime_contract": {
                    "required_receipts": [],
                    "required_upstream_stages": [{"stage": "index", "layouts": ["production"]}],
                }
            },
        },
    )
    generic = rl.bind_lineage(
        workspace, run_id, "extract", registry_path=registry,
        receipt_path_for_stage=lambda stage_id: None, layout=rl.LAYOUT_GENERIC,
    )
    assert generic["upstream_receipts"] == {}
    assert generic["binding_source"]["layout"] == "generic"
    receipt = {"mode": "live", "script_registry_sha256": registry_sha256(registry), "required_script_ids": [],
               "lineage": generic, "output_digest": "o"}
    passed = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="extract", receipt=receipt,
        registry_path=registry, layout=rl.LAYOUT_GENERIC,
    )
    assert passed.status == "pass", passed.blockers
    production = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="extract", receipt=receipt, registry_path=registry,
    )
    assert "stage_contract_lineage_upstream_unbound" in [blocker["code"] for blocker in production.blockers]

    bound = rl.bind_lineage(
        workspace, run_id, "extract", registry_path=registry,
        receipt_path_for_stage=rl.receipt_path_resolver_from_registry(workspace, run_dir, registry_path=registry),
    )
    assert bound["upstream_receipts"][f"storage/workflow-runs/{run_id}/stage-06-index.json"]["stage_id"] == "index"


def test_registry_rejects_malformed_layout_scoped_upstream_entries(tmp_path: Path) -> None:
    import pytest

    from conference_pipeline.stage_script_registry import StageScriptRegistryError, load_stage_script_registry

    for entry in (
        {"stage": "index"},
        {"stage": "index", "layouts": []},
        {"stage": "index", "layouts": ["nightly"]},
        {"stage": "index", "layouts": ["production"], "optional": True},
        {"layouts": ["production"]},
    ):
        registry = _registry(
            tmp_path / "bad.json",
            {"index": {}, "extract": {"runtime_contract": {"required_upstream_stages": [entry]}}},
        )
        with pytest.raises(StageScriptRegistryError, match="required_upstream_stages"):
            load_stage_script_registry(registry)


def test_every_generic_required_upstream_stage_is_a_generic_workflow_stage() -> None:
    """The generic workflow can only bind stages it writes receipts for."""

    from conference_pipeline import workflow
    from conference_pipeline.stage_script_registry import load_stage_script_registry

    registry = load_stage_script_registry()
    generic_ids = {stage.id for stage in workflow.WORKFLOW_STAGES}
    unresolvable: dict[str, list[str]] = {}
    for stage_id, binding in registry["stage_bindings"].items():
        if stage_id not in generic_ids:
            continue
        contracts = [binding.get("runtime_contract") or {}] + [
            workspace_binding.get("runtime_contract") or {}
            for workspace_binding in (binding.get("workspace_bindings") or {}).values()
        ]
        for contract in contracts:
            missing = [
                upstream
                for upstream in rl.required_upstream_stage_ids(contract, layout=rl.LAYOUT_GENERIC)
                if upstream not in generic_ids
            ]
            if missing:
                unresolvable[stage_id] = missing
    assert unresolvable == {}
    extraction = registry["stage_bindings"]["exhaustive-topic-extraction"]["runtime_contract"]
    assert rl.required_upstream_stage_ids(extraction, layout=rl.LAYOUT_PRODUCTION) == ["corpus-inventory-index"]


def test_canonical_output_sha256_ignores_run_time_stamps_at_any_depth(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _write_json(first, {"generated_at": "t1", "deck": {"taken_at": "t1", "slides": 3}, "approved_at": "a1"})
    _write_json(second, {"generated_at": "t2", "deck": {"taken_at": "t2", "slides": 3}, "approved_at": "a1"})
    assert rl.canonical_output_sha256(first) == rl.canonical_output_sha256(second)
    _write_json(second, {"generated_at": "t2", "deck": {"taken_at": "t2", "slides": 3}, "approved_at": "a2"})
    assert rl.canonical_output_sha256(first) != rl.canonical_output_sha256(second)  # a new approval is real
    text = tmp_path / "notes.md"
    text.write_text("x", encoding="utf-8")
    assert rl.canonical_output_sha256(text) is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert rl.canonical_output_sha256(tmp_path / "bad.json") is None


def test_upstream_freshness_blockers_flag_a_stale_or_missing_upstream_before_any_work(tmp_path: Path) -> None:
    """The pre-spend gate reports what the final verification would, before the
    stage runs: a direct upstream always matches a fresh binding, so only an
    upstream that is itself stale, lacks an output_digest, or is missing blocks."""

    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    registry = _registry(
        tmp_path / "registry.json",
        {
            "root": {"receipt_filename": "stage-01-root.json", "runtime_contract": {}},
            "middle": {
                "receipt_filename": "stage-02-middle.json",
                "runtime_contract": {"required_upstream_stages": ["root"]},
            },
            "leaf": {
                "receipt_filename": "stage-03-leaf.json",
                "runtime_contract": {"required_upstream_stages": ["middle"]},
            },
        },
    )
    resolver = rl.receipt_path_resolver_from_registry(workspace, run_dir, registry_path=registry)

    def gate(stage_id: str = "leaf") -> list[dict]:
        return rl.upstream_freshness_blockers(
            workspace, "run-1", stage_id, receipt_path_for_stage=resolver, registry_path=registry
        )

    # Missing required upstream.
    assert [b["code"] for b in gate()] == ["stage_contract_upstream_changed"]

    root = run_dir / "stage-01-root.json"
    middle = run_dir / "stage-02-middle.json"
    _write_json(root, {"stage_id": "root", "stage_order": 1, "status": "succeeded", "output_digest": "r1", "lineage": {"upstream_receipts": {}}})
    middle_lineage = {
        "upstream_receipts": {
            "storage/workflow-runs/run-1/stage-01-root.json": {"stage_id": "root", "digest_kind": "output_digest", "digest": "r1"}
        }
    }
    _write_json(middle, {"stage_id": "middle", "stage_order": 2, "status": "succeeded", "output_digest": "m1", "lineage": middle_lineage})
    assert gate() == []

    # Root re-ran: middle is stale, so leaf must not run yet; middle itself may.
    _write_json(root, {"stage_id": "root", "stage_order": 1, "status": "succeeded", "output_digest": "r2", "lineage": {"upstream_receipts": {}}})
    blockers = gate()
    assert [(b["code"], b["upstream_stage_id"]) for b in blockers] == [("stage_contract_upstream_stale", "middle")]
    assert gate("middle") == []
    summary = rl.summarize_lineage_blockers(blockers)
    assert summary[0]["chain"] == ["storage/workflow-runs/run-1/stage-02-middle.json"]
    assert "upstream_blockers" not in summary[0]

    # A stage upstream without output_digest (pre-v5.1) blocks too.
    _write_json(middle, {"stage_id": "middle", "stage_order": 2, "status": "succeeded", "lineage": middle_lineage})
    assert [b["code"] for b in gate()] == ["stage_contract_upstream_output_digest_missing"]


def _two_stage_run_with_outputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    registry = _registry(
        tmp_path / "registry.json",
        {
            "root": {"receipt_filename": "stage-01-root.json", "runtime_contract": {}},
            "leaf": {
                "receipt_filename": "stage-02-leaf.json",
                "runtime_contract": {"required_upstream_stages": ["root"]},
            },
        },
    )
    out = workspace / "analysis"
    out.mkdir(parents=True)
    (out / "summary.md").write_text("measured", encoding="utf-8")
    (out / "report.md").write_text("provided", encoding="utf-8")
    (out / "selection.json").write_text('{"keep": 1, "extra": 2}', encoding="utf-8")
    manifest = {"content_sha256": "", "topics": [1]}
    import hashlib

    manifest["content_sha256"] = hashlib.sha256(
        json.dumps({"content_sha256": "", "topics": [1]}, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    _write_json(out / "manifest.json", {**manifest, "generated_at": "t0"})
    artifacts = rl.digest_output_artifacts(
        workspace,
        [
            {"name": "summary", "path": "analysis/summary.md"},
            {"name": "report", "path": "analysis/report.md", "sha256": rl.sha256_file(out / "report.md")},
            # A stage-specific scheme no generic digest reproduces.
            {"name": "selection", "path": "analysis/selection.json", "sha256": "f" * 64},
            {"name": "manifest", "path": "analysis/manifest.json", "content_sha256": manifest["content_sha256"]},
            {"name": "offloaded", "path": "analysis/drive-only.json", "sha256": "e" * 64},
        ],
    )
    root = run_dir / "stage-01-root.json"
    _write_json(
        root,
        {
            "stage_id": "root",
            "stage_order": 1,
            "status": "succeeded",
            "output_artifacts": artifacts,
            "output_digest": rl.compute_output_digest(artifacts),
            "lineage": {"upstream_receipts": {}},
        },
    )
    return workspace, registry, root


def test_upstream_freshness_blockers_rehash_an_upstream_stage_outputs_on_disk(tmp_path: Path) -> None:
    """An upstream output edited after its receipt was written blocks the paid
    downstream stage even though every receipt digest still matches."""

    import os

    workspace, registry, root = _two_stage_run_with_outputs(tmp_path)
    run_dir = root.parent
    resolver = rl.receipt_path_resolver_from_registry(workspace, run_dir, registry_path=registry)

    def gate() -> list[dict]:
        return rl.upstream_freshness_blockers(
            workspace, "run-1", "leaf", receipt_path_for_stage=resolver, registry_path=registry
        )

    # Unknown-scheme, self-attested and Drive-offloaded outputs are not edits.
    assert gate() == []
    assert rl.stage_lineage_report(workspace, "run-1", layout="production", registry_path=registry)["stages"]["root"]["state"] == "fresh"

    later = root.stat().st_mtime + 10
    # Rewriting the manifest with the same content (new generated_at) is no edit.
    manifest = json.loads((workspace / "analysis" / "manifest.json").read_text(encoding="utf-8"))
    _write_json(workspace / "analysis" / "manifest.json", {**manifest, "generated_at": "t1"})
    assert gate() == []

    # A measured output is compared exactly; a provided one too once it is newer than the receipt.
    (workspace / "analysis" / "summary.md").write_text("edited", encoding="utf-8")
    (workspace / "analysis" / "report.md").write_text("edited", encoding="utf-8")
    os.utime(workspace / "analysis" / "report.md", (later, later))
    blockers = gate()
    assert [(b["code"], b["upstream_stage_id"]) for b in blockers] == [
        ("stage_contract_upstream_output_changed", "root")
    ]
    assert blockers[0]["paths"] == ["analysis/summary.md", "analysis/report.md"]
    assert rl.summarize_lineage_blockers(blockers)[0]["paths"] == ["analysis/summary.md", "analysis/report.md"]

    # An unknown-scheme output written after the receipt counts as edited.
    os.utime(workspace / "analysis" / "selection.json", (later, later))
    assert "analysis/selection.json" in gate()[0]["paths"]

    # stage-status shows the edited stage itself as stale, with the paths.
    report = rl.stage_lineage_report(workspace, "run-1", layout="production", registry_path=registry)
    entry = report["stages"]["root"]
    assert entry["state"] == "stale"
    assert entry["reasons"][0]["code"] == "stage_contract_output_changed"
    assert "analysis/summary.md" in entry["reasons"][0]["paths"]
    assert report["next_stage_to_run"] == "root"


def test_changed_output_artifacts_ignores_receipts_without_outputs(tmp_path: Path) -> None:
    assert rl.changed_output_artifacts(tmp_path, {"stage_id": "x"}) == []
    assert rl.changed_output_artifacts(tmp_path, {"output_artifacts": [{"name": "a", "path": "a", "sha256": None, "missing": True}]}) == []


def test_upstream_freshness_blockers_report_an_unresolvable_binding(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    workspace.mkdir(parents=True)
    registry = _registry(
        tmp_path / "registry.json",
        {"stage": {"workspace_bindings": {"conferences/other2026": {}}, "runtime_contract": {}}},
    )
    blockers = rl.upstream_freshness_blockers(workspace, "run-1", "stage", registry_path=registry)
    assert [b["code"] for b in blockers] == ["stage_contract_binding_unresolved"]


def _substep_registry(tmp_path: Path) -> Path:
    registry = _registry(
        tmp_path / "registry.json",
        {
            "index": {"receipt_filename": "stage-06-index.json", "runtime_contract": {}},
            "analysis": {
                "receipt_filename": "stage-07-analysis.json",
                "runtime_contract": {
                    "required_receipts": [{"path": "storage/workflow-runs/<run-id>/07B-select.json"}],
                },
            },
        },
    )
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["lineage_substeps"] = {
        "full-read": {
            "receipt": "storage/workflow-runs/<run-id>/14A-full-read.json",
            "required_upstream_stages": ["index"],
            "required_substeps": [],
        },
        "select": {
            "receipt": "storage/workflow-runs/<run-id>/07B-select.json",
            "required_upstream_stages": [],
            "required_substeps": ["full-read"],
        },
    }
    _write_json(registry, payload)
    return registry


def test_substep_receipts_carry_lineage_and_a_rebuilt_stage_makes_their_consumers_stale(tmp_path: Path) -> None:
    """A sub-step receipt bound by content alone was a leaf: rebuilding the
    corpus index could never reach the stage consuming it. Declared sub-steps
    now stamp lineage and the verifier follows it; one without lineage is stale."""

    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    registry = _substep_registry(tmp_path)
    resolver = rl.receipt_path_resolver_from_registry(workspace, run_dir, registry_path=registry)

    def gate() -> list[tuple[str, str | None]]:
        blockers = rl.upstream_freshness_blockers(
            workspace, "run-1", "analysis", receipt_path_for_stage=resolver, registry_path=registry
        )
        return [(b["code"], b.get("upstream_stage_id")) for b in blockers]

    _write_json(run_dir / "stage-06-index.json", {"stage_id": "index", "stage_order": 6, "status": "succeeded", "output_digest": "i1", "lineage": {"upstream_receipts": {}}})
    # Pre-v5.1 sub-step receipts (no lineage, no output digest) block, like a
    # stage receipt written before lineage existed; they are never trusted leaves.
    _write_json(run_dir / "14A-full-read.json", {"status": "succeeded", "units": 3})
    _write_json(run_dir / "07B-select.json", {"status": "succeeded", "questions": 2})
    assert gate() == [("stage_contract_upstream_output_digest_missing", "select")]

    # Writers stamp lineage when they write; the chain then verifies.
    for name, substep in (("14A-full-read.json", "full-read"), ("07B-select.json", "select")):
        path = run_dir / name
        receipt = json.loads(path.read_text(encoding="utf-8"))
        rl.stamp_substep_receipt(receipt, workspace=workspace, run_id="run-1", substep_id=substep, registry_path=registry)
        _write_json(path, receipt)
    full_read = json.loads((run_dir / "14A-full-read.json").read_text(encoding="utf-8"))
    assert list(full_read["lineage"]["upstream_receipts"]) == ["storage/workflow-runs/run-1/stage-06-index.json"]
    assert gate() == []

    # The corpus index is rebuilt: the full read, then the selection, then analysis are stale.
    _write_json(run_dir / "stage-06-index.json", {"stage_id": "index", "stage_order": 6, "status": "succeeded", "output_digest": "i2", "lineage": {"upstream_receipts": {}}})
    blockers = rl.upstream_freshness_blockers(
        workspace, "run-1", "analysis", receipt_path_for_stage=resolver, registry_path=registry
    )
    assert [(b["code"], b["upstream_stage_id"]) for b in blockers] == [("stage_contract_upstream_stale", "select")]
    inner = blockers[0]["upstream_blockers"]
    assert [(b["code"], b["upstream_stage_id"]) for b in inner] == [("stage_contract_upstream_stale", "full-read")]

    report = rl.stage_lineage_report(workspace, "run-1", layout="production", registry_path=registry)
    assert {key: entry["state"] for key, entry in report["substeps"].items()} == {"full-read": "stale", "select": "stale"}
    assert list(report["substeps"]) == ["full-read", "select"]
    assert report["next_substep_to_run"] == "full-read"


def test_stamp_substep_receipt_never_raises_and_leaves_an_unbound_receipt(tmp_path: Path) -> None:
    registry = _substep_registry(tmp_path)
    receipt: dict = {"status": "succeeded", "lineage": {"stale": True}}
    rl.stamp_substep_receipt(receipt, workspace=tmp_path / "ws", run_id="run-1", substep_id="undeclared", registry_path=registry)
    assert "lineage" not in receipt
    assert receipt["lineage_error"].startswith("KeyError")


def test_content_digest_of_a_substep_ignores_its_lineage(tmp_path: Path) -> None:
    # A downstream binds a sub-step by content; stamping lineage must not move it.
    assert rl.content_digest({"status": "succeeded", "n": 1}) == rl.content_digest(
        {"status": "succeeded", "n": 1, "lineage": {"upstream_receipts": {}}}
    )


def test_registry_rejects_malformed_lineage_substeps(tmp_path: Path) -> None:
    import pytest

    from conference_pipeline.stage_script_registry import StageScriptRegistryError, load_stage_script_registry

    base = json.loads(_substep_registry(tmp_path).read_text(encoding="utf-8"))
    cases = {
        "unknown_substep": {"a": {"receipt": "x/a.json", "required_substeps": ["nope"]}},
        "unknown_stage": {"a": {"receipt": "x/a.json", "required_upstream_stages": ["nope"]}},
        "cycle": {
            "a": {"receipt": "x/a.json", "required_substeps": ["b"]},
            "b": {"receipt": "x/b.json", "required_substeps": ["a"]},
        },
        "duplicate_receipt": {"a": {"receipt": "x/a.json"}, "b": {"receipt": "x/a.json"}},
        "no_receipt": {"a": {"required_substeps": []}},
        "bad_outputs": {"a": {"receipt": "x/a.json", "outputs": "analysis/x"}},
    }
    for name, substeps in cases.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, {**base, "lineage_substeps": substeps})
        with pytest.raises(StageScriptRegistryError):
            load_stage_script_registry(path)


def test_every_declared_substep_is_stamped_by_a_writer() -> None:
    """Each registry sub-step has a writer calling stamp_substep_receipt with its id."""

    import re

    from conference_pipeline.stage_script_registry import load_stage_script_registry

    src = Path(__file__).resolve().parents[1] / "src" / "conference_pipeline"
    stamped: set[str] = set()
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        stamped.update(re.findall(r'stamp_substep_receipt\([^)]*substep_id="([^"]+)"', text, flags=re.S))
    declared = set(rl.substep_specs(load_stage_script_registry()))
    assert declared == stamped, (declared - stamped, stamped - declared)


def test_substeps_bind_by_output_digest_so_run_bookkeeping_never_invalidates(tmp_path: Path) -> None:
    """An unchanged resume rewrites a sub-step receipt's timings and reuse
    counts; consumers bind its ``substep_outputs`` digest, so they stay fresh.
    A changed output file does move it. ``refresh_substep_outputs`` re-digests
    in place (sanctioned repairs) and keeps the recorded lineage."""

    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    registry = _substep_registry(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["lineage_substeps"]["full-read"]["outputs"] = ["out/full-read"]
    _write_json(registry, payload)
    _write_json(workspace / "out" / "full-read" / "u1.json", {"summary": "a", "generated_at": "t1", "reused": False})
    _write_json(run_dir / "stage-06-index.json", {"stage_id": "index", "stage_order": 6, "status": "succeeded", "output_digest": "i1", "lineage": {"upstream_receipts": {}}})

    def write_full_read(extra: dict) -> None:
        receipt = {"status": "succeeded", "checks": {"elapsed_seconds": 1.0}, **extra}
        rl.stamp_substep_receipt(receipt, workspace=workspace, run_id="run-1", substep_id="full-read", registry_path=registry)
        _write_json(run_dir / "14A-full-read.json", receipt)

    write_full_read({})
    select = rl.stamp_substep_receipt(
        {"status": "succeeded"}, workspace=workspace, run_id="run-1", substep_id="select", registry_path=registry
    )
    entry = next(iter(select["lineage"]["upstream_receipts"].values()))
    assert entry["digest_kind"] == "output_digest"

    def blockers() -> list[str]:
        return [b["code"] for b in rl.verify_lineage(workspace, "run-1", select["lineage"], stage_id="select", registry_path=registry)]

    assert blockers() == []
    # Unchanged resume: new timings, reuse bookkeeping and stamps only.
    _write_json(workspace / "out" / "full-read" / "u1.json", {"summary": "a", "generated_at": "t2", "reused": True})
    write_full_read({"checks": {"elapsed_seconds": 9.5, "chars_per_second": 3.0}, "skipped_existing_count": 1})
    assert blockers() == []
    # A real output change moves the digest.
    _write_json(workspace / "out" / "full-read" / "u1.json", {"summary": "b", "generated_at": "t2"})
    write_full_read({})
    assert blockers() == ["stage_contract_upstream_changed"]

    # refresh_substep_outputs: re-digests in place, keeps lineage untouched.
    before = json.loads((run_dir / "14A-full-read.json").read_text(encoding="utf-8"))
    _write_json(workspace / "out" / "full-read" / "u1.json", {"summary": "repaired"})
    assert rl.refresh_substep_outputs(workspace, "run-1", "full-read", registry_path=registry) is True
    after = json.loads((run_dir / "14A-full-read.json").read_text(encoding="utf-8"))
    assert after["lineage"] == before["lineage"]
    assert after["substep_outputs"]["digest"] != before["substep_outputs"]["digest"]
    assert rl.refresh_substep_outputs(workspace, "run-1", "select", registry_path=registry) is False  # no receipt


def test_only_a_succeeded_complete_substep_is_bound_or_counts_as_current(tmp_path: Path) -> None:
    """A dry run, failed attempt or partial-scope re-run must not bind current
    lineage over outputs it did not recompute, and verification rejects a
    sub-step upstream that is not succeeded."""

    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    registry = _substep_registry(tmp_path)
    _write_json(run_dir / "stage-06-index.json", {"stage_id": "index", "stage_order": 6, "status": "succeeded", "output_digest": "i1", "lineage": {"upstream_receipts": {}}})

    planned = rl.stamp_substep_receipt(
        {"status": "planned", "lineage": {"old": True}}, workspace=workspace, run_id="run-1", substep_id="full-read", registry_path=registry
    )
    assert "lineage" not in planned and planned["lineage_unbound_reason"] == "status:planned"
    partial = rl.stamp_substep_receipt(
        {"status": "succeeded"}, workspace=workspace, run_id="run-1", substep_id="full-read", complete=False, registry_path=registry
    )
    assert "lineage" not in partial and partial["lineage_unbound_reason"] == "partial_scope"

    full = rl.stamp_substep_receipt(
        {"status": "succeeded"}, workspace=workspace, run_id="run-1", substep_id="full-read", registry_path=registry
    )
    _write_json(run_dir / "14A-full-read.json", full)
    select = rl.stamp_substep_receipt(
        {"status": "succeeded"}, workspace=workspace, run_id="run-1", substep_id="select", registry_path=registry
    )
    assert rl.verify_lineage(workspace, "run-1", select["lineage"], stage_id="select", registry_path=registry) == []
    # A later dry run rewrites the full-read receipt as planned (same outputs).
    _write_json(run_dir / "14A-full-read.json", {**full, "status": "planned"})
    codes = [b["code"] for b in rl.verify_lineage(workspace, "run-1", select["lineage"], stage_id="select", registry_path=registry)]
    assert codes == ["stage_contract_upstream_not_succeeded"]


def test_sanitize_receipt_for_write_keeps_lineage_verifiable(tmp_path: Path) -> None:
    """A receipt scrubbed on its way to disk must still verify.

    The receipt sanitizer redacts by key name, and lineage records
    ``exclude_keys``: the field names a projected input digest skips.  Scrubbing
    those turned them into ``"REDACTED"``, so verification projected a digest
    over the wrong keys and reported ``input_artifact_changed`` for an input
    nothing had touched -- the 2026-09-20 CVPR full read, an 11-hour run.
    """

    workspace = tmp_path / "ws"
    (workspace / "analysis").mkdir(parents=True)
    source = workspace / "analysis" / "topic_strategy.json"
    source.write_text(
        json.dumps({"topics": [{"id": "t1"}], "provider_receipt_paths": ["a.json"]}),
        encoding="utf-8",
    )
    projected = rl._artifact_sha256(source, exclude_keys=["provider_receipt_paths"])

    receipt = {
        "status": "succeeded",
        "api_key": "sk-test_ABCDEFGHIJKLMNOP",
        "lineage": {
            "input_artifacts": {
                "topic_strategy": {
                    "path": "analysis/topic_strategy.json",
                    "sha256": projected,
                    "exclude_keys": ["provider_receipt_paths"],
                    "optional": True,
                    "missing": False,
                    "source": "registry",
                }
            }
        },
    }

    written = rl.sanitize_receipt_for_write(receipt)

    # The credential is still redacted; the lineage reaches disk untouched.
    assert written["api_key"] == "REDACTED"
    assert written["lineage"] == receipt["lineage"]
    assert (
        rl.verify_input_artifacts(workspace, written["lineage"], stage_id="llm-full-read") == []
    )

    # The same receipt scrubbed wholesale is what regressed: it verifies stale.
    from conference_pipeline.iclr2025_live import sanitize_for_receipt

    scrubbed = sanitize_for_receipt(dict(receipt))
    scrubbed["lineage"]["input_artifacts"]["topic_strategy"]["exclude_keys"] = ["REDACTED"]
    codes = [
        blocker["code"]
        for blocker in rl.verify_input_artifacts(
            workspace, scrubbed["lineage"], stage_id="llm-full-read"
        )
    ]
    assert codes == ["stage_contract_input_artifact_changed"]
