"""Shared Stage 4 (download-corpus) frame.

Stage 4 used to live in every conference runner. These tests pin what the
shared frame does for ANY runner: lane ordering (acquire before gate, in
declaration order), paid-lane authorization, blocker resolution and
completion, capacity, prior-inventory carry-forward, immutable Stage 4
attempt receipts, the pinned program-item lineage, and a minimal new
conference (the ``minimal_new_conference`` fixture plus a second synthetic
slug) running the frozen-corpus format with **no lanes at all**, then with
one synthetic lane. Nothing touches the network or a provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conference_pipeline.production import ProductionRunnerContext, stage_table
from conference_pipeline.production import download_corpus_stage as frame
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production.hooks import (
    LANE_PHASE_GATE,
    NO_HOOKS,
    AcquisitionLane,
    ConferenceHooks,
    LaneResult,
    validate_hooks,
)
from conference_pipeline.provider_receipts import utc_now
from conference_pipeline.workflow_runs import validate_run_id
from tests.orchestrator_golden import materialize_fixture_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "conference_pipeline" / "production" / "download_corpus_stage.py"
RUN_ID = "newconf-stage4"
STAGES = stage_table.from_registry(
    next_actions={
        "corpus-freeze": "Repair the corpus-freeze blockers, then rerun corpus-freeze.",
        "download-corpus": "Repair the download-corpus blockers, then rerun download-corpus.",
    },
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _context(workspace: Path, run_id: str = RUN_ID) -> ProductionRunnerContext:
    slug = workspace.name
    return ProductionRunnerContext(
        workspace=workspace,
        run_id=run_id,
        stages=STAGES,
        runner_id=slug,
        runner_path=f"scripts/run_{slug}_production.py",
        receipt_type=f"conference_pipeline.{slug}_production_stage_receipt",
        repo_root=REPO_ROOT,
    )


def _second_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "otherconf2033"
    (workspace / "storage").mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        'conference:\n  slug: "otherconf2033"\n  name: "Other Conference 2033"\n  year: 2033\n', encoding="utf-8"
    )
    return workspace


@pytest.fixture(params=["minimal_new_conference", "otherconf2033"])
def ctx(request: pytest.FixtureRequest, tmp_path: Path) -> ProductionRunnerContext:
    if request.param == "minimal_new_conference":
        workspace = materialize_fixture_workspace("minimal_new_conference", tmp_path / "conferences")
    else:
        workspace = _second_workspace(tmp_path)
    return _context(workspace)


def run_conventions(ctx: ProductionRunnerContext, written_blockers: list[dict[str, Any]]) -> production_orchestrator.RunConventions:
    def build_receipt(*, workspace: Path, run_id: str, stage: Any, mode: str, status: str, started_at: str,
                      checks: dict[str, Any], error: str | None = None, blocker_receipt_path: str | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "receipt_type": ctx.receipt_type,
            "run_id": run_id,
            "workspace": str(workspace),
            "stage_id": stage.id,
            "stage_order": stage.order,
            "mode": mode,
            "status": status,
            "started_at": started_at,
            "checks": checks,
            "error": error,
            "blocker_receipt_path": blocker_receipt_path,
        }

    def write_blocker(**kwargs: Any) -> Path:
        written_blockers.append(kwargs)
        return _write_json(ctx.run_dir() / "blocker-receipts" / f"{kwargs['stage_id']}-{len(written_blockers)}.json", kwargs)

    return production_orchestrator.RunConventions(
        build_receipt=build_receipt,
        write_receipt=lambda workspace, run_id, stage, receipt: _write_json(ctx.receipt_path(stage.id), receipt),
        write_summary=lambda **kwargs: production_orchestrator.write_run_summary(
            summary_path=ctx.run_dir() / "summary.json",
            generated_at=utc_now(),
            unblock_receipts=[],
            write_json=_write_json,
            **kwargs,
        ),
        active_blockers=lambda *_a, **_k: [],
        context=lambda workspace, run_id: _context(workspace, run_id),
        run_dir=lambda workspace, run_id: workspace / "storage" / "workflow-runs" / validate_run_id(run_id),
        now=utc_now,
        validate_run_id=validate_run_id,
        write_blocker=write_blocker,
    )


def _settings(slug: str) -> frame.FrozenCorpusSettings:
    return frame.FrozenCorpusSettings(
        inventory_receipt_type=f"conference_pipeline.{slug}_download_corpus",
        report_title=f"{slug} Download Corpus Report",
        load_env=lambda _env: {"attempted": False, "exists": False, "loaded": False},
        cost_approval=lambda _args: {"approved": False},
    )


def _freeze(workspace: Path, *, status: str = "succeeded") -> Path:
    return _write_json(
        workspace / "storage" / "corpus-freeze" / "corpus_freeze.json",
        {
            "status": status,
            "conference": workspace.name,
            "blockers": [],
            "included_records": [
                {"source_family_id": "papers", "source_url": "https://example.invalid/papers"},
            ],
        },
    )


def _args(workspace: Path) -> SimpleNamespace:
    return SimpleNamespace(workspace=workspace, run_id=RUN_ID, mode="live", env_file=None)


# --------------------------------------------------------------------------
# Module boundaries
# --------------------------------------------------------------------------


def test_frame_module_carries_no_conference_literal() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8").lower()
    for token in ("icml", "cvpr", "iclr", "neurips", "2026"):
        assert token not in source, token


# --------------------------------------------------------------------------
# Lane ordering, authorization, resolution, completion
# --------------------------------------------------------------------------


def _recording_lane(lane_id: str, calls: list[str], **kwargs: Any) -> AcquisitionLane:
    result = kwargs.pop("result", LaneResult())

    def run(_ctx: Any, _frozen: Any, _prior: Any) -> LaneResult:
        calls.append(lane_id)
        return result

    return AcquisitionLane(lane_id=lane_id, run=run, **kwargs)


def test_lanes_run_in_declaration_order_and_gates_run_only_in_their_phase() -> None:
    calls: list[str] = []
    lanes = (
        _recording_lane("gate-a", calls, phase=LANE_PHASE_GATE),
        _recording_lane("first", calls),
        _recording_lane("second", calls),
    )
    blockers: list[dict[str, Any]] = []
    acquire = frame.run_acquisition_lanes(lanes, ctx=None, frozen_corpus={}, prior_inventory=None, blockers=blockers, phase="acquire")
    gate = frame.run_acquisition_lanes(lanes, ctx=None, frozen_corpus={}, prior_inventory=None, blockers=blockers, phase="gate")
    assert calls == ["first", "second", "gate-a"]
    assert [record.lane_id for record in acquire] == ["first", "second"]
    assert [record.lane_id for record in gate] == ["gate-a"]


def test_a_paid_lane_runs_only_with_preflight_authorization() -> None:
    calls: list[str] = []
    lane = _recording_lane("paid", calls, paid=True, result=LaneResult(blockers=[{"code": "x"}]))
    blockers: list[dict[str, Any]] = []
    skipped = frame.run_acquisition_lanes((lane,), ctx=None, frozen_corpus={}, prior_inventory=None, blockers=blockers)
    assert calls == [] and blockers == []
    assert skipped[0].status == frame.UNAUTHORIZED_PAID_LANE and skipped[0].result.completed is False
    ran = frame.run_acquisition_lanes(
        (lane,), ctx=None, frozen_corpus={}, prior_inventory=None, blockers=blockers, authorize_paid=lambda _lane: True
    )
    assert calls == ["paid"] and ran[0].status == "ran" and blockers == [{"code": "x"}]


def test_a_lane_resolves_an_earlier_pending_blocker_after_adding_its_own() -> None:
    lanes = (
        _recording_lane("talks", [], result=LaneResult(blockers=[{"code": "pending_escalation"}, {"code": "kept"}])),
        _recording_lane("escalation", [], result=LaneResult(blockers=[{"code": "final"}], resolves=("pending_escalation",))),
    )
    blockers: list[dict[str, Any]] = []
    frame.run_acquisition_lanes(lanes, ctx=None, frozen_corpus={}, prior_inventory=None, blockers=blockers)
    assert [blocker["code"] for blocker in blockers] == ["kept", "final"]


def test_incomplete_lanes_record_their_declared_blocker_in_lane_order() -> None:
    lanes = (
        _recording_lane("a", [], incomplete_blocker="a_incomplete", result=LaneResult(completed=False)),
        _recording_lane("b", [], incomplete_blocker="b_incomplete"),
        _recording_lane("c", [], incomplete_blocker="c_incomplete", paid=True),
        _recording_lane("d", []),
    )
    records = frame.run_acquisition_lanes(lanes, ctx=None, frozen_corpus={}, prior_inventory=None, blockers=[])
    codes = [blocker["code"] for blocker in frame.incomplete_lane_blockers(lanes, records)]
    assert codes == ["a_incomplete", "c_incomplete"]


def test_a_lane_must_return_a_lane_result() -> None:
    lane = AcquisitionLane(lane_id="bad", run=lambda *_a: {"status": "ok"})
    with pytest.raises(TypeError, match="LaneResult"):
        frame.run_acquisition_lanes((lane,), ctx=None, frozen_corpus={}, prior_inventory=None, blockers=[])


def test_validate_hooks_rejects_an_unknown_lane_phase() -> None:
    lane = AcquisitionLane(lane_id="odd", run=lambda *_a: LaneResult(), phase="later")
    problems = validate_hooks(ConferenceHooks(acquisition_lanes=(lane,)), ["download-corpus"])
    assert any("unknown phase" in problem for problem in problems)


def test_select_lane_picks_the_first_enabled_exclusive_route() -> None:
    routes = (("registration", False), ("events", True), ("other", True))
    assert frame.select_lane(routes, default="public") == "events"
    assert frame.select_lane((("registration", False),), default="public") == "public"


def test_run_bounded_queues_bounded_batches_and_persists_between_them() -> None:
    seen: list[int] = []
    batches: list[int] = []
    frame.run_bounded(
        list(range(10)),
        lambda job: job * 2,
        worker_count=2,
        batch_factor=2,
        on_result=seen.append,
        after_batch=lambda: batches.append(len(seen)),
    )
    assert sorted(seen) == [value * 2 for value in range(10)]
    assert batches == [4, 8, 10]


# --------------------------------------------------------------------------
# Prior-inventory merge and capacity
# --------------------------------------------------------------------------


def test_carry_forward_copies_only_accepted_prior_artifacts(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "download_corpus.json",
        {"direct_download_artifacts": [{"id": 1, "keep": True}, {"id": 2}, "junk", {"id": 3, "keep": True}]},
    )
    prior = frame.load_prior_inventory(path)
    carried = frame.carry_forward_prior_artifacts(prior, lambda artifact: bool(artifact.get("keep")))
    assert carried == [{"id": 1, "keep": True}, {"id": 3, "keep": True}]
    assert carried[0] is not prior["direct_download_artifacts"][0]
    assert frame.load_prior_inventory(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert frame.load_prior_inventory(tmp_path / "bad.json") is None


@pytest.mark.parametrize(
    ("available", "status", "codes"),
    [
        (None, "blocked_unknown", ["disk_capacity_unknown"]),
        (10, "blocked_insufficient", ["disk_capacity_insufficient"]),
        (500, "pass_pre_request", []),
    ],
)
def test_capacity_is_reserved_before_any_request(available: Any, status: str, codes: list[str]) -> None:
    capacity, blockers = frame.evaluate_capacity({"required_bytes": 100}, available)
    assert capacity["status"] == status
    assert [blocker["code"] for blocker in blockers] == codes
    assert all(blocker["affected_stage"] == "download-corpus" for blocker in blockers)


def test_storage_controls_require_a_bounded_safety_margin(tmp_path: Path) -> None:
    from conference_pipeline.production.corpus_freeze_stage import read_json_object

    manifest, margin, blockers = frame.read_storage_controls(tmp_path, read_json_object=read_json_object)
    assert manifest is None and margin == 0 and [b["code"] for b in blockers] == ["storage_manifest_missing"]
    _write_json(tmp_path / frame.STORAGE_MANIFEST, {"families": []})
    _write_json(tmp_path / frame.STORAGE_POLICY, {"safety_margin_bytes": -1})
    manifest, margin, blockers = frame.read_storage_controls(tmp_path, read_json_object=read_json_object)
    assert manifest == {"families": []} and margin == 0
    assert [b["code"] for b in blockers] == ["storage_policy_safety_margin_invalid"]


# --------------------------------------------------------------------------
# Format 1: a minimal new conference, no lanes at all
# --------------------------------------------------------------------------


def test_minimal_conference_with_no_lanes_blocks_on_the_missing_freeze(ctx: ProductionRunnerContext) -> None:
    blockers: list[dict[str, Any]] = []
    exit_code, summary = frame.run_frozen_corpus_stage(
        run_conventions(ctx, blockers), _args(ctx.workspace), STAGES["download-corpus"], settings=_settings(ctx.slug)
    )
    receipt = _read_json(ctx.receipt_path("download-corpus"))
    assert exit_code == 1 and summary["status"] == "blocked"
    assert receipt["checks"]["corpus_freeze_present"] is False
    assert blockers[0]["remediation_path"] == "rerun corpus-freeze before download-corpus"


def test_minimal_conference_with_no_lanes_runs_the_frame_and_blocks_honestly(ctx: ProductionRunnerContext) -> None:
    _freeze(ctx.workspace)
    blockers: list[dict[str, Any]] = []
    exit_code, _summary = frame.run_frozen_corpus_stage(
        run_conventions(ctx, blockers), _args(ctx.workspace), STAGES["download-corpus"], settings=_settings(ctx.slug), hooks=NO_HOOKS
    )
    receipt = _read_json(ctx.receipt_path("download-corpus"))
    inventory = _read_json(ctx.workspace / frame.DOWNLOAD_CORPUS_INVENTORY)
    assert exit_code == 1
    assert inventory["receipt_type"] == f"conference_pipeline.{ctx.slug}_download_corpus"
    assert [blocker["code"] for blocker in inventory["blockers"]] == ["paper_pdf_downloads_missing"]
    assert inventory["local_artifact_count"] == 0 and inventory["discovered_target_count"] == 0
    assert receipt["checks"]["paper_artifact_count"] == 0
    report = (ctx.workspace / frame.DOWNLOAD_CORPUS_REPORT).read_text(encoding="utf-8")
    assert report.startswith(f"# {ctx.slug} Download Corpus Report")


def test_minimal_conference_with_one_synthetic_lane_succeeds(ctx: ProductionRunnerContext) -> None:
    _freeze(ctx.workspace)
    pdf = ctx.workspace / "papers" / "synthetic.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    seen_auth: list[Any] = []

    def lane(lane_ctx: frame.LaneContext, frozen: dict[str, Any], prior: Any) -> LaneResult:
        seen_auth.append(lane_ctx.auth.credentials.get("token"))
        assert frozen["included_records"] and prior is None
        artifact = {"artifact_kind": "paper_pdf", "status": "succeeded", "local_path": "papers/synthetic.pdf"}
        return LaneResult(artifacts=[artifact], receipt_fields={"synthetic_lane": {"status": "ran"}})

    hooks = ConferenceHooks(
        auth=lambda _args: frame.AuthPosture(
            check_fields={"site_session": {"loaded": True}}, credentials={"token": "never-recorded"}
        ),
        acquisition_lanes=(AcquisitionLane(lane_id="synthetic-papers", run=lane),),
    )
    settings = frame.FrozenCorpusSettings(**{**_settings(ctx.slug).__dict__, "check_payload_keys": ("synthetic_lane",)})
    exit_code, summary = frame.run_frozen_corpus_stage(
        run_conventions(ctx, []), _args(ctx.workspace), STAGES["download-corpus"], settings=settings, hooks=hooks
    )
    receipt = _read_json(ctx.receipt_path("download-corpus"))
    inventory_text = (ctx.workspace / frame.DOWNLOAD_CORPUS_INVENTORY).read_text(encoding="utf-8")
    assert exit_code == 0 and summary["status"] == "succeeded"
    assert seen_auth == ["never-recorded"] and "never-recorded" not in inventory_text
    assert receipt["checks"]["site_session"] == {"loaded": True}
    assert receipt["checks"]["synthetic_lane"] == {"status": "ran"}
    assert receipt["checks"]["paper_artifact_count"] == 1
    assert json.loads(inventory_text)["direct_download_artifact_count"] == 1


def test_a_resumed_stage_hands_the_prior_inventory_to_its_lanes(ctx: ProductionRunnerContext) -> None:
    _freeze(ctx.workspace)
    _write_json(ctx.workspace / frame.DOWNLOAD_CORPUS_INVENTORY, {"direct_download_artifacts": [{"artifact_kind": "paper_pdf", "status": "succeeded", "kept": True}]})

    def carry(_ctx: Any, _frozen: Any, prior: Any) -> LaneResult:
        return LaneResult(artifacts=frame.carry_forward_prior_artifacts(prior, lambda artifact: bool(artifact.get("kept"))))

    exit_code, _ = frame.run_frozen_corpus_stage(
        run_conventions(ctx, []),
        _args(ctx.workspace),
        STAGES["download-corpus"],
        settings=_settings(ctx.slug),
        hooks=ConferenceHooks(acquisition_lanes=(AcquisitionLane(lane_id="carry", run=carry),)),
    )
    inventory = _read_json(ctx.workspace / frame.DOWNLOAD_CORPUS_INVENTORY)
    assert exit_code == 0
    assert inventory["direct_download_artifacts"] == [{"artifact_kind": "paper_pdf", "kept": True, "status": "succeeded"}]


# --------------------------------------------------------------------------
# Format 2: immutable attempts and the pinned program lineage
# --------------------------------------------------------------------------


class _Contract:
    def __init__(self, status: str) -> None:
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status}


def _ledger_conventions(ctx: ProductionRunnerContext, *, contract: str = "pass", lineage_blockers: list[dict[str, Any]] | None = None) -> frame.LedgerConventions:
    from conference_pipeline.production.corpus_freeze_stage import read_json_object

    return frame.LedgerConventions(
        stages=STAGES,
        run_dir=lambda workspace, run_id: workspace / "storage" / "workflow-runs" / run_id,
        now=utc_now,
        write_json=lambda path, payload: _write_json(path, payload),
        write_bytes=lambda path, payload: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(payload)),
        finalize_receipt=lambda receipt, **_kwargs: receipt,
        verify_contract=lambda **_kwargs: _Contract(contract),
        read_json_object=read_json_object,
        receipt_type=ctx.receipt_type,
        owner_id=f"acquisition.download_corpus.{ctx.slug}",
        validate_source_lineage=lambda **_kwargs: ({}, {}, list(lineage_blockers or [])),
    )


def test_stage4_attempt_receipts_are_immutable(ctx: ProductionRunnerContext) -> None:
    conventions = _ledger_conventions(ctx)
    common = dict(workspace=ctx.workspace, run_id=RUN_ID, body={"downloads_made": 0}, checks={}, stage_contract=None)
    first, _, _ = frame.write_stage_receipt(conventions, status="blocked", blockers=[{"code": "one"}], **common)
    second, _, _ = frame.write_stage_receipt(conventions, status="blocked", blockers=[{"code": "two"}], **common)
    third, receipt, summary = frame.write_stage_receipt(conventions, status="succeeded", blockers=[], **common)
    run_dir = ctx.workspace / "storage" / "workflow-runs" / RUN_ID
    assert first == third == run_dir / STAGES["download-corpus"].receipt_filename
    assert ".attempt-" in second.name and _read_json(second)["blockers"] == [{"code": "two"}]
    superseded = Path(receipt["superseded_prior_receipt_path"])
    assert ".superseded-" in superseded.name and _read_json(superseded)["blockers"] == [{"code": "one"}]
    assert summary["status"] == "succeeded" and receipt["required_script_ids"] == [conventions.owner_id]


def test_program_ledger_stage_blocks_a_foreign_run_before_any_lane(ctx: ProductionRunnerContext) -> None:
    calls: list[str] = []
    outcome = frame.run_program_ledger_stage(
        _ledger_conventions(ctx, contract="fail"),
        workspace=ctx.workspace,
        run_id=RUN_ID,
        settings=frame.ProgramLedgerSettings(
            execution_status_key="download_corpus_execution_status",
            owner_function="run_download_corpus_stage",
            report_title="Synthetic Download Corpus Report",
        ),
        lanes=(_recording_lane("public", calls, incomplete_blocker="public_incomplete"),),
        pinned_run_id="some-other-run",
        pinned_inventory_sha256="0" * 64,
        expected_program_item_count=1,
        cap_hits=("--max-items",),
    )
    codes = [blocker["code"] for blocker in outcome["receipt"]["blockers"]]
    assert calls == []
    assert outcome["status"] == "blocked" and outcome["ledger"] is None
    assert codes == ["pinned_source_run_mismatch", "silent_item_cap_rejected", "final_stage_contract_incomplete"]
    assert outcome["receipt"]["checks"]["download_corpus_execution_status"] == "blocked_pre_request"
