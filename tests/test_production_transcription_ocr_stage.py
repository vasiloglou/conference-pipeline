"""Shared Stage 5 (transcription-ocr) module.

Stage 5 used to live in every conference runner. These tests pin what the
shared module does for ANY runner: a minimal new conference (the
``minimal_new_conference`` fixture, plus a second synthetic slug) runs the
default local lanes with no hooks under its own receipt writers, and a
declared transcription lane (with and without a retry authorization) runs on
a pinned Stage 4 handoff with fake lanes and verifiers. Nothing touches the
network or a provider.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conference_pipeline.production import (
    ProductionRunnerContext,
    stage_table,
    transcription_ocr_stage,
)
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production.hooks import NO_HOOKS, ConferenceHooks
from conference_pipeline.production.receipts import StageIO
from conference_pipeline.provider_receipts import utc_now
from conference_pipeline.receipt_lineage import sha256_file
from conference_pipeline.transcription_ocr import (
    TRANSCRIPTION_OCR_OWNER_ID,
    TRANSCRIPTION_OCR_OWNER_REFS,
)
from conference_pipeline.workflow_runs import validate_run_id
from tests.orchestrator_golden import materialize_fixture_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "conference_pipeline" / "production" / "transcription_ocr_stage.py"
RUN_ID = "newconf-stage5"
STAGES = stage_table.from_registry(
    next_actions={
        "download-corpus": "Repair the download-corpus blockers, then rerun download-corpus.",
        "transcription-ocr": "Repair the transcription-ocr blockers, then rerun transcription-ocr.",
    },
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


IO = StageIO(write_json=lambda path, payload: _write_json(path, payload))


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
    def build_receipt(
        *,
        workspace: Path,
        run_id: str,
        stage: Any,
        mode: str,
        status: str,
        started_at: str,
        checks: dict[str, Any],
        error: str | None = None,
        blocker_receipt_path: str | None = None,
    ) -> dict[str, Any]:
        receipt = {
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
            "required_script_ids": ["runner.own_binding"],
            "error": error,
            "blocker_receipt_path": blocker_receipt_path,
        }
        return ctx.finalize(receipt, stage_id=stage.id)

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


def run_local(ctx: ProductionRunnerContext, blockers: list[dict[str, Any]], **overrides: Any) -> tuple[int, dict[str, Any]]:
    options: dict[str, Any] = {
        "workspace": ctx.workspace,
        "run_id": ctx.run_id,
        "stage": ctx.stage("transcription-ocr"),
        "upstream_stage": ctx.stage("download-corpus"),
        "mode": "live",
        "env_file_status": lambda: {"attempted": False, "exists": False, "loaded": False},
        "cost_approval": lambda: {"approved": False, "approval_sources": []},
        "worker_count": 1,
        "resume": False,
        "io": IO,
    }
    options.update(overrides)
    return transcription_ocr_stage.run_local_lanes_stage(run_conventions(ctx, blockers), **options)


def seed_download(ctx: ProductionRunnerContext, *, inventory: dict[str, Any] | None = None) -> None:
    _write_json(ctx.receipt_path("download-corpus"), {"stage_id": "download-corpus", "status": "succeeded"})
    _write_json(
        ctx.workspace / transcription_ocr_stage.DOWNLOAD_CORPUS_INVENTORY,
        inventory
        if inventory is not None
        else {"stage_id": "download-corpus", "status": "succeeded", "local_artifacts": [], "direct_download_artifacts": []},
    )


def _receipt(ctx: ProductionRunnerContext) -> dict[str, Any]:
    return json.loads(ctx.receipt_path("transcription-ocr").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Format 1: the default local lanes, no hooks
# --------------------------------------------------------------------------


def test_missing_download_receipt_blocks_naming_the_shared_lane_owner(ctx: ProductionRunnerContext) -> None:
    blockers: list[dict[str, Any]] = []

    exit_code, summary = run_local(ctx, blockers)

    assert exit_code == 1 and summary["status"] == "blocked"
    receipt = _receipt(ctx)
    assert receipt["receipt_type"] == ctx.receipt_type
    assert receipt["required_script_ids"] == [TRANSCRIPTION_OCR_OWNER_ID]
    assert receipt["stage_script_owner_refs"] == list(TRANSCRIPTION_OCR_OWNER_REFS)
    assert receipt["checks"]["download_corpus_stage_receipt_present"] is False
    assert receipt["checks"]["provider_calls_made"] is False
    assert receipt["error"].startswith("missing download-corpus stage receipt")
    assert blockers[0]["remediation_path"] == "rerun download-corpus before transcription-ocr"
    assert {key: receipt["lineage"]["code_identity"][key] for key in ctx.code_identity} == ctx.code_identity


def test_unsucceeded_inventory_and_malformed_waivers_block(ctx: ProductionRunnerContext) -> None:
    seed_download(ctx, inventory={"stage_id": "download-corpus", "status": "blocked"})
    exit_code, _summary = run_local(ctx, [])
    assert exit_code == 1
    assert _receipt(ctx)["error"] == "transcription-ocr requires a succeeded download-corpus inventory"

    seed_download(ctx)
    _write_json(ctx.workspace / transcription_ocr_stage.OPERATOR_WAIVERS, [])
    exit_code, _summary = run_local(ctx, [])
    assert exit_code == 1
    receipt = _receipt(ctx)
    assert receipt["error"].startswith("transcription/OCR operator waiver file is malformed")
    assert receipt["checks"]["operator_waivers_present"] is True


def test_local_lanes_run_with_no_hooks_and_succeed(ctx: ProductionRunnerContext) -> None:
    seed_download(ctx)
    _write_json(
        ctx.workspace / transcription_ocr_stage.OPERATOR_WAIVERS,
        {
            "schema_version": 1,
            "waivers": [
                {
                    "code": "no_local_audio_video_inputs",
                    "decision": "waived",
                    "reason": "fixture conference publishes no talk media",
                    "operator": "fixture",
                }
            ],
        },
    )

    exit_code, summary = run_local(ctx, [])

    assert exit_code == 0, summary
    receipt = _receipt(ctx)
    assert receipt["status"] == "succeeded"
    assert receipt["required_script_ids"] == [TRANSCRIPTION_OCR_OWNER_ID]
    assert receipt["checks"]["provider_cost_approval"] == {"approved": False, "approval_sources": []}
    assert (ctx.workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json").is_file()


def test_lane_blockers_block_with_the_inventory_remediation(ctx: ProductionRunnerContext) -> None:
    seed_download(ctx)
    calls: list[dict[str, Any]] = []
    approvals = iter(({"approved": True}, {"approved": True, "approval_sources": ["fixture"]}))

    def fake_lanes(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(checks={"lane": "fake"}, blockers=[{"code": "b_code"}, {"code": "a_code"}])

    blockers: list[dict[str, Any]] = []
    exit_code, _summary = run_local(
        ctx,
        blockers,
        run_lanes=fake_lanes,
        cost_approval=lambda: next(approvals),
        worker_count=0,
        speech=transcription_ocr_stage.SpeechTranscriptionOptions(run_provider=True, max_rpm=0),
    )

    assert exit_code == 1
    assert calls[0]["worker_count"] == 1 and calls[0]["speech_max_rpm"] == 1
    assert calls[0]["run_speech_provider"] is True and calls[0]["speech_provider_cost_approved"] is True
    receipt = _receipt(ctx)
    assert receipt["error"] == "transcription-ocr blocked: a_code, b_code"
    assert receipt["checks"]["lane"] == "fake"
    assert receipt["checks"]["provider_cost_approval"]["approval_sources"] == ["fixture"]
    assert "transcription_ocr_inventory.json" in blockers[0]["remediation_path"]


def test_talk_media_gate_uses_the_settings_wording(ctx: ProductionRunnerContext, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_download(ctx)
    monkeypatch.setattr(
        transcription_ocr_stage,
        "workshop_talk_media_input_binding",
        lambda **_kwargs: {"workshop_talk_media_input_blockers": [{"code": "workshop_talk_media_acquisition_required"}]},
    )
    settings = transcription_ocr_stage.LocalLanesSettings(talk_media_reason="needs talks", talk_media_remediation="get talks")
    blockers: list[dict[str, Any]] = []

    exit_code, _summary = run_local(ctx, blockers, settings=settings)

    assert exit_code == 1
    assert _receipt(ctx)["error"] == "needs talks"
    assert blockers[0]["remediation_path"] == "get talks"
    assert blockers[0]["blocker_class"] == "missing_receipt"


# --------------------------------------------------------------------------
# Format 2: a declared transcription lane on a pinned Stage 4 handoff
# --------------------------------------------------------------------------

LANE_SETTINGS = transcription_ocr_stage.LaneStageSettings(
    owner_id="transcription_ocr.fixture_lane",
    owner_function="run_fixture_lane",
    receipt_type="conference_pipeline.fixture_transcription_ocr_stage_receipt",
    execution_status_key="transcription_ocr_fixture_execution_status",
    inventory_path=Path("storage/transcription-ocr/fixture/inventory.json"),
    processor_identity="fixture-processor",
    processor_version="1",
)


class Contract:
    def __init__(self, status: str) -> None:
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "blockers": [] if self.status == "pass" else [{"code": "fixture"}]}


def fixture_lane(ctx: ProductionRunnerContext, events: list[str], *, fail: bool = False, reuse: bool = False) -> transcription_ocr_stage.TranscriptionLane:
    inventory_path = ctx.workspace / LANE_SETTINGS.inventory_path

    def build_inventory(*, workspace: Path, receipt_path: Path) -> dict[str, Any]:
        events.append("build")
        assert receipt_path == ctx.receipt_path("download-corpus")
        if fail:
            raise ValueError("handoff not admissible")
        if reuse:
            # A compatible prior is returned in memory and never rewritten.
            prior = json.loads(inventory_path.read_text(encoding="utf-8"))
            return {**prior, "resume": {"reused_output_digest": prior["inventory_sha256"]}}
        inventory = {"inventory_sha256": "c" * 64, "resume": {}}
        _write_json(inventory_path, inventory)
        return inventory

    def checks(*, workspace: Path, run_id: str, inventory: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        values: dict[str, Any] = {LANE_SETTINGS.execution_status_key: "blocked"}
        if inventory is None:
            return values, []
        values["transcription_ocr_inventory_sha256"] = sha256_file(inventory_path)
        values["transcription_ocr_inventory_canonical_sha256"] = inventory["inventory_sha256"]
        return values, []

    def resume_is_exact(*, prior: Any, inventory: Any, inventory_file_sha256: str) -> bool:
        return bool(prior) and prior.get("status") == "succeeded" and prior["checks"]["transcription_ocr_inventory_sha256"] == inventory_file_sha256

    return transcription_ocr_stage.TranscriptionLane(
        lane_id="fixture", build_inventory=build_inventory, checks=checks, resume_is_exact=resume_is_exact
    )


def lane_owners(status: str = "pass") -> transcription_ocr_stage.LaneStageOwners:
    return transcription_ocr_stage.LaneStageOwners(verify_contract=lambda **_kwargs: Contract(status), io=IO)


def test_lane_stage_requires_exactly_one_declared_lane(ctx: ProductionRunnerContext) -> None:
    with pytest.raises(ValueError, match="exactly one declared TranscriptionLane"):
        transcription_ocr_stage.run_lane_stage(ctx, settings=LANE_SETTINGS, hooks=NO_HOOKS)


def test_lane_stage_succeeds_and_resume_reuses_the_exact_result(ctx: ProductionRunnerContext) -> None:
    events: list[str] = []
    hooks = ConferenceHooks(transcription_lanes=(fixture_lane(ctx, events),))

    outcome = transcription_ocr_stage.run_lane_stage(ctx, settings=LANE_SETTINGS, hooks=hooks, owners=lane_owners())

    assert outcome["status"] == "succeeded"
    receipt = _receipt(ctx)
    assert receipt["receipt_type"] == LANE_SETTINGS.receipt_type
    assert receipt["required_script_ids"] == [LANE_SETTINGS.owner_id]
    assert receipt["checks"][LANE_SETTINGS.execution_status_key] == "pass"
    assert {key: receipt["lineage"]["code_identity"][key] for key in ctx.code_identity} == ctx.code_identity
    before = ctx.receipt_path("transcription-ocr").read_bytes()

    resumed_hooks = ConferenceHooks(transcription_lanes=(fixture_lane(ctx, events, reuse=True),))
    resumed = transcription_ocr_stage.run_lane_stage(
        ctx, settings=LANE_SETTINGS, hooks=resumed_hooks, resume=True, owners=lane_owners()
    )

    assert resumed["summary"] == {"status": "succeeded", "reused_exact_result": True}
    assert ctx.receipt_path("transcription-ocr").read_bytes() == before


def test_strict_contract_failure_never_publishes_a_pass(ctx: ProductionRunnerContext) -> None:
    hooks = ConferenceHooks(transcription_lanes=(fixture_lane(ctx, []),))

    outcome = transcription_ocr_stage.run_lane_stage(ctx, settings=LANE_SETTINGS, hooks=hooks, owners=lane_owners("blocked"))

    assert outcome["status"] == "blocked"
    receipt = _receipt(ctx)
    assert receipt["checks"][LANE_SETTINGS.execution_status_key] == "blocked"
    assert {"code": "final_transcription_ocr_stage_contract_incomplete"} in receipt["blockers"]


def test_default_verifier_fails_closed_without_a_registry_binding(ctx: ProductionRunnerContext) -> None:
    hooks = ConferenceHooks(transcription_lanes=(fixture_lane(ctx, []),))

    outcome = transcription_ocr_stage.run_lane_stage(
        ctx, settings=LANE_SETTINGS, hooks=hooks, owners=transcription_ocr_stage.LaneStageOwners(io=IO)
    )

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["stage_contract"]["status"] != "pass"


def test_admission_failure_after_a_success_writes_an_attempt_receipt(ctx: ProductionRunnerContext) -> None:
    transcription_ocr_stage.run_lane_stage(
        ctx, settings=LANE_SETTINGS, hooks=ConferenceHooks(transcription_lanes=(fixture_lane(ctx, []),)), owners=lane_owners()
    )
    canonical = ctx.receipt_path("transcription-ocr")
    before = canonical.read_bytes()

    outcome = transcription_ocr_stage.run_lane_stage(
        ctx,
        settings=LANE_SETTINGS,
        hooks=ConferenceHooks(transcription_lanes=(fixture_lane(ctx, [], fail=True),)),
        owners=lane_owners(),
    )

    assert outcome["status"] == "blocked"
    assert Path(outcome["receipt_path"]) != canonical and ".attempt-" in outcome["receipt_path"]
    assert outcome["receipt"]["blockers"][-1] == {"code": "transcription_ocr_local_admission_failed", "reason": "ValueError"}
    assert canonical.read_bytes() == before


def test_pinned_lane_refuses_other_runs(ctx: ProductionRunnerContext) -> None:
    lane = fixture_lane(ctx, [])
    pinned = transcription_ocr_stage.TranscriptionLane(
        lane_id=lane.lane_id,
        build_inventory=lane.build_inventory,
        checks=lane.checks,
        resume_is_exact=lane.resume_is_exact,
        pinned_run_id="another-run",
        pinned_run_error="fixture lane is pinned",
    )
    with pytest.raises(ValueError, match="fixture lane is pinned"):
        transcription_ocr_stage.run_lane_stage(ctx, settings=LANE_SETTINGS, hooks=ConferenceHooks(transcription_lanes=(pinned,)))


def test_retry_authorization_without_a_hook_is_refused_before_the_lane(ctx: ProductionRunnerContext, tmp_path: Path) -> None:
    events: list[str] = []
    with pytest.raises(ValueError, match="no transcription-ocr retry authorization"):
        transcription_ocr_stage.run_lane_stage(
            ctx,
            settings=LANE_SETTINGS,
            hooks=ConferenceHooks(transcription_lanes=(fixture_lane(ctx, events),)),
            retry_authorization=tmp_path / "marker.json",
        )
    assert events == []


def _retry_hook(ctx: ProductionRunnerContext, events: list[str], *, valid: bool = True) -> transcription_ocr_stage.RetryAuthorization:
    control = ctx.run_dir() / "retry-attempt.json"

    def claim(*, workspace: Path, run_id: str, retry_authorization: Path) -> tuple[dict[str, Any], dict[str, Path]]:
        events.append("claim")
        transcription_ocr_stage.write_json_exclusive_and_fsync(control, {"status": "claimed"})
        return {"status": "claimed"}, {"attempt": control}

    def record_attempt(*, path: Path, claimed: Any, status: str, details: Any) -> dict[str, Any]:
        events.append(f"record:{status}")
        return {}

    def finalize_controls(**_kwargs: Any) -> dict[str, dict[str, str]]:
        events.append("finalize")
        return {"attempt": {"path": "storage/attempt.json", "sha256": sha256_file(control)}}

    def validate_controls(*, workspace: Path, run_id: str, candidate: Any) -> dict[str, dict[str, str]] | None:
        events.append("validate")
        return dict(candidate) if valid and candidate else None

    return transcription_ocr_stage.RetryAuthorization(
        claim=claim, record_attempt=record_attempt, finalize_controls=finalize_controls, validate_controls=validate_controls
    )


@pytest.mark.parametrize("valid", [True, False], ids=["controls-hold", "controls-invalid"])
def test_retry_claim_precedes_the_lane_and_controls_are_revalidated(
    ctx: ProductionRunnerContext, tmp_path: Path, valid: bool
) -> None:
    events: list[str] = []
    lane = fixture_lane(ctx, events)
    hooks = ConferenceHooks(transcription_lanes=(lane,), retry_authorization=_retry_hook(ctx, events, valid=valid))

    outcome = transcription_ocr_stage.run_lane_stage(
        ctx, settings=LANE_SETTINGS, hooks=hooks, retry_authorization=tmp_path / "marker.json", owners=lane_owners()
    )

    assert events[:2] == ["claim", "build"]
    if valid:
        assert outcome["status"] == "succeeded"
        assert set(outcome["receipt"]["checks"][transcription_ocr_stage.RETRY_CONTROLS_KEY]) == {"attempt"}
    else:
        assert outcome["status"] == "blocked"
        assert transcription_ocr_stage.RETRY_CONTROLS_KEY not in outcome["receipt"]["checks"]
        assert {"code": "transcription_ocr_retry_control_finalization_failed"} in outcome["receipt"]["blockers"]


def test_durable_controls_never_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "controls" / "marker.json"
    transcription_ocr_stage.write_json_exclusive_and_fsync(path, {"a": 1})
    with pytest.raises(FileExistsError):
        transcription_ocr_stage.write_json_exclusive_and_fsync(path, {"a": 2})
    transcription_ocr_stage.replace_json_and_fsync(path, {"a": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 3}
    assert transcription_ocr_stage.is_sha256(transcription_ocr_stage.json_digest({"a": 3}))
    assert transcription_ocr_stage.load_receipt(tmp_path / "missing.json") is None
    paths = transcription_ocr_stage.control_paths(tmp_path / "run", {"b": "b.json", "a": "a.json"})
    assert list(paths) == ["b", "a"]
    assert transcription_ocr_stage.relative_paths(tmp_path, paths) == {"b": "run/b.json", "a": "run/a.json"}


def test_module_carries_no_conference_literals() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8").lower()
    for token in ("neurips", "iclr", "icml", "cvpr", "r5", "r4"):
        assert re.search(rf"\b{token}", source) is None, token
