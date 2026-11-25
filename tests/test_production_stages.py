"""Shared production stages (``conference_pipeline.production``).

Stages 11 and 12 used to exist only inside the CVPR 2026 runner, with the
conference name written into the proof labels and the receipt type repeated as
a literal. These tests pin what the shared module must do for ANY runner: take
its identity from the context, write receipts in the runner's own type, and
refuse exactly what the CVPR-local code refused.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from conference_pipeline.production import ProductionRunnerContext
from conference_pipeline.production import proof_stage, release_stage
from conference_pipeline.production.decks import presentation_deck_scope

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Stage:
    id: str
    order: int
    receipt_filename: str
    implemented: bool
    next_action: str


STAGES = {
    "analysis": _Stage("analysis", 7, "stage-07-analysis.json", True, "rerun analysis"),
    "presentation": _Stage("presentation", 10, "stage-10-presentation.json", True, "rerun presentation"),
    "notebooklm-podcast": _Stage("notebooklm-podcast", 10, "stage-10-notebooklm-podcast.json", True, "rerun podcasts"),
    "release": _Stage("release", 11, "stage-11-release.json", True, "rerun release"),
    "proof": _Stage("proof", 12, "stage-12-proof.json", True, "rerun proof"),
}


def _workspace(tmp_path: Path, slug: str) -> Path:
    workspace = tmp_path / "conferences" / slug
    workspace.mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        "\n".join(
            [
                "conference:",
                f'  slug: "{slug}"',
                f'  name: "Test Conference {slug}"',
                "  year: 2026",
                "urls:",
                '  base: "https://example.test"',
                '  virtual: "https://example.test/virtual/2026"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (workspace / "storage").mkdir()
    (workspace / "storage" / "storage_manifest.json").write_text(json.dumps({"families": {}}), encoding="utf-8")
    (workspace / "analysis").mkdir()
    (workspace / "analysis" / "analysis_manifest.json").write_text(
        json.dumps({"topics": [{"id": "01_alpha_topic", "title": "Alpha"}]}), encoding="utf-8"
    )
    return workspace


def _context(workspace: Path, run_id: str = "test-shared-run", **overrides: object) -> ProductionRunnerContext:
    params: dict[str, object] = dict(
        workspace=workspace,
        run_id=run_id,
        stages=STAGES,
        runner_id="testrunner",
        runner_path="scripts/run_test_production.py",
        receipt_type="conference_pipeline.testrunner_production_stage_receipt",
        repo_root=REPO_ROOT,
    )
    params.update(overrides)
    return ProductionRunnerContext(**params)  # type: ignore[arg-type]


def _write_receipt(ctx: ProductionRunnerContext, stage_id: str, status: str) -> None:
    ctx.run_dir().mkdir(parents=True, exist_ok=True)
    ctx.receipt_path(stage_id).write_text(
        json.dumps({"stage_id": stage_id, "status": status, "mode": "live", "stage_contract": {"status": "pass"}}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def test_context_takes_identity_from_the_runner_and_slug_from_the_workspace(tmp_path: Path) -> None:
    ctx = _context(_workspace(tmp_path, "test2026"))
    assert ctx.code_identity == {"runner": "testrunner", "runner_path": "scripts/run_test_production.py"}
    assert ctx.slug == "test2026"
    assert ctx.conference_display_name() == "Test Conference test2026"
    assert ctx.run_dir() == (ctx.workspace / "storage" / "workflow-runs" / "test-shared-run").resolve()
    assert ctx.receipt_path("release").name == "stage-11-release.json"
    with pytest.raises(KeyError):
        ctx.stage("no-such-stage")


def test_context_falls_back_to_the_directory_name_without_a_config(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "bare2026"
    workspace.mkdir(parents=True)
    ctx = _context(workspace)
    assert ctx.slug == "bare2026"
    assert ctx.conference_display_name(default="") == ""


def test_deck_scope_lists_manifest_topics_then_the_runner_executive_decks() -> None:
    scope = presentation_deck_scope(
        {"topics": [{"id": "01_a", "title": "A"}, {"id": "02_b", "title": "B"}]},
        (("executive_summary_short", "Short"),),
    )
    assert [(row["id"], row["kind"]) for row in scope] == [
        ("01_a", "topic"),
        ("02_b", "topic"),
        ("executive_summary_short", "executive"),
    ]
    with pytest.raises(RuntimeError, match="has_no_topics"):
        presentation_deck_scope({"topics": []}, ())


# ---------------------------------------------------------------------------
# Stage 11 release
# ---------------------------------------------------------------------------


def test_release_deliverables_follow_the_context_layout(tmp_path: Path) -> None:
    ctx = _context(
        _workspace(tmp_path, "test2026"),
        podcast_dir=Path("podcasts"),
        notebooklm_subdir="nlm",
        notebooklm_audio_filename="episode.m4a",
    )
    items = release_stage.release_deliverables(ctx, ["01_alpha_topic"])
    by_id = {row["id"]: row for row in items}
    assert by_id["01_alpha_topic:podcast_notebooklm"]["path"] == "podcasts/01_alpha_topic/nlm/episode.m4a"
    assert by_id["01_alpha_topic:podcast_notebooklm"]["produced_by"] == "notebooklm-podcast"
    assert by_id["01_alpha_topic:pdf"]["path"] == "decks/01_alpha_topic/presentation.pdf"
    assert by_id["analysis_manifest"]["produced_by"] == "analysis"
    # The storage manifest is a contract every stage reads, never a deliverable.
    assert not any(row["path"].endswith("storage_manifest.json") for row in items)


def test_release_blocks_over_a_presentation_stage_that_did_not_succeed(tmp_path: Path) -> None:
    ctx = _context(_workspace(tmp_path, "test2026"))
    _write_receipt(ctx, "presentation", "blocked")
    outcome = release_stage.run_release_stage(ctx, resume=False)
    assert outcome["status"] == "blocked"
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["release_presentation_stage_not_succeeded"]
    receipt = json.loads(ctx.receipt_path("release").read_text(encoding="utf-8"))
    assert receipt["receipt_type"] == "conference_pipeline.testrunner_production_stage_receipt"
    assert receipt["lineage"]["code_identity"]["runner"] == "testrunner"
    assert receipt["stage_order"] == 11
    assert receipt["next_action"] == "rerun release"
    assert outcome["summary"]["blockers"] == ["release_presentation_stage_not_succeeded"]


def test_release_assembles_an_inventory_and_keeps_publication_blocked_without_approval(tmp_path: Path) -> None:
    ctx = _context(_workspace(tmp_path, "test2026"))
    _write_receipt(ctx, "analysis", "succeeded")
    _write_receipt(ctx, "presentation", "succeeded")
    _write_receipt(ctx, "notebooklm-podcast", "succeeded")
    outcome = release_stage.run_release_stage(ctx, resume=False, publish_target="drive", publish_approved_by=None)
    inventory = json.loads((ctx.run_dir() / release_stage.RELEASE_INVENTORY_FILENAME).read_text(encoding="utf-8"))
    assert inventory["conference_name"] == "Test Conference test2026"
    assert inventory["publication"]["blocked"] is True
    assert {row["stage_id"] for row in inventory["provenance"]} == {"analysis", "presentation", "notebooklm-podcast"}
    checks = outcome["receipt"]["checks"]
    assert checks["deck_count"] == 3  # one topic + two default executive decks
    assert checks["release_bundle_sha256"]
    assert checks["owner_execution_evidence"]["release.bundle"].startswith(str(ctx.run_dir()))
    # Nothing was produced, so the deliverable digests block the release honestly.
    assert outcome["status"] == "blocked"
    assert "release_final_stage_contract_failed" in outcome["summary"]["blockers"]


# ---------------------------------------------------------------------------
# Stage 12 proof
# ---------------------------------------------------------------------------


def test_proof_labels_are_derived_from_the_workspace_slug() -> None:
    candidates = proof_stage.support_label_candidates("test2026")
    assert {row["proposed_label"] for row in candidates} == {"test2026-live-proven"}
    assert any(row["requires_stages"] == ["notebooklm-podcast"] for row in candidates)
    assert any(row["requires_stages"] == ["release"] for row in candidates)


def test_proof_blocks_over_a_release_stage_that_did_not_succeed(tmp_path: Path) -> None:
    ctx = _context(_workspace(tmp_path, "test2026"))
    outcome = proof_stage.run_proof_stage(ctx, resume=False)
    assert outcome["status"] == "blocked"
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["proof_release_stage_not_succeeded"]
    assert outcome["receipt"]["checks"]["release_status"] is None
    assert ctx.receipt_path("proof").is_file()


def test_proof_blocks_when_the_release_inventory_is_missing(tmp_path: Path) -> None:
    ctx = _context(_workspace(tmp_path, "test2026"))
    _write_receipt(ctx, "release", "succeeded")
    outcome = proof_stage.run_proof_stage(ctx, resume=False)
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["proof_release_inventory_missing"]


# ---------------------------------------------------------------------------
# Runner adoption: ICML 2026 dispatches release/proof through the module
# ---------------------------------------------------------------------------


def _load_icml_runner():
    runner_path = REPO_ROOT / "scripts" / "run_icml2026_production.py"
    spec = importlib.util.spec_from_file_location("icml2026_runner_for_production_stages", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_icml_runner_declares_release_and_proof_as_implemented() -> None:
    runner = _load_icml_runner()
    assert runner.STAGES["release"].implemented is True
    assert runner.STAGES["proof"].implemented is True
    args = runner.parse_args(["--stage", "release", "--publish-target", "drive", "--publish-approved-by", "op"])
    assert (args.publish_target, args.publish_approved_by) == ("drive", "op")


def test_icml_runner_release_writes_its_own_receipt_type_and_a_run_summary(tmp_path: Path) -> None:
    runner = _load_icml_runner()
    workspace = _workspace(tmp_path, "icml2026")
    args = runner.parse_args(["--stage", "release", "--workspace", str(workspace), "--run-id", "icml-shared-test"])
    exit_code, summary = runner.main(
        ["--stage", "release", "--workspace", str(workspace), "--run-id", "icml-shared-test", "--summary"]
    ), None
    assert exit_code == 1  # no presentation receipt: blocked, never "unavailable"
    receipt = json.loads(
        (workspace / "storage" / "workflow-runs" / "icml-shared-test" / "stage-11-release.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["receipt_type"] == "conference_pipeline.icml2026_production_stage_receipt"
    assert receipt["lineage"]["code_identity"]["runner"] == "icml2026"
    assert [item["code"] for item in receipt["blockers"]] == ["release_presentation_stage_not_succeeded"]
    run_summary = json.loads(
        (workspace / "storage" / "workflow-runs" / "icml-shared-test" / "summary.json").read_text(encoding="utf-8")
    )
    assert run_summary["status"] == "blocked"
    assert run_summary["halt_stage_id"] == "release"
    assert run_summary["stages"][0]["stage_id"] == "release"
    # No STAGE_UNAVAILABLE blocker receipt is written any more.
    assert runner.active_blockers(workspace, run_id="icml-shared-test") == []
    del args


# ---------------------------------------------------------------------------
# Stage 10.4 NotebookLM podcast (ported from the CVPR runner tests, which can
# no longer run on main; the shared stage is what both runners call)
# ---------------------------------------------------------------------------

PODCAST_STAGES = {
    **STAGES,
    "qa-gap-fill-synthesis": _Stage(
        "qa-gap-fill-synthesis", 8, "stage-08-qa-gap-fill-synthesis.json", True, "rerun synthesis"
    ),
}


def _podcast_context(workspace: Path, run_id: str, **overrides: object) -> ProductionRunnerContext:
    return _context(workspace, run_id, stages=PODCAST_STAGES, **overrides)


def _podcast_workspace(tmp_path: Path) -> Path:
    from tests.test_podcast_notebooklm import make_workspace

    return make_workspace(tmp_path, with_executive=True, with_executive_long=True)


def _seed_podcast_predecessors(ctx: ProductionRunnerContext) -> None:
    from tests.lineage_fixtures import stamp_upstream_stage_receipts

    run_dir = ctx.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx.receipt_path("qa-gap-fill-synthesis").write_text(
        json.dumps(
            {
                "stage_id": "qa-gap-fill-synthesis",
                "status": "succeeded",
                "mode": "live",
                "checks": {"executive_summary_status": "pass", "trend_report_status": "pass"},
            }
        ),
        encoding="utf-8",
    )
    ctx.receipt_path("presentation").write_text(
        json.dumps({"stage_id": "presentation", "status": "succeeded", "mode": "live", "stage_contract": {"status": "pass"}}),
        encoding="utf-8",
    )
    stamp_upstream_stage_receipts(run_dir)


def _passing_transcriber():
    from tests.test_podcast_notebooklm import passing_transcriber

    return passing_transcriber()


def test_notebooklm_blocks_on_missing_inputs_without_network(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn
    from conference_pipeline.production import notebooklm_stage

    ctx = _podcast_context(_podcast_workspace(tmp_path), "nb-missing")
    engine = pn.FakeNotebookLMEngine()
    outcome = notebooklm_stage.run_notebooklm_podcast_stage(
        ctx, ack_live_notebooklm=True, engine=engine, transcriber=_passing_transcriber()
    )
    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["blockers"][0]["code"] == "notebooklm_podcast_input_receipt_missing"
    assert engine.calls == []
    assert outcome["receipt"]["receipt_type"] == "conference_pipeline.testrunner_production_stage_receipt"


def test_notebooklm_blocks_without_live_ack(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn
    from conference_pipeline.production import notebooklm_stage

    ctx = _podcast_context(_podcast_workspace(tmp_path), "nb-no-ack")
    _seed_podcast_predecessors(ctx)
    engine = pn.FakeNotebookLMEngine()
    outcome = notebooklm_stage.run_notebooklm_podcast_stage(ctx, engine=engine)
    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["blockers"][0]["code"] == "notebooklm_live_ack_required"
    assert engine.calls == []
    assert outcome["receipt"]["provider_calls_made"] is False


def test_notebooklm_deck_ids_canary_is_always_blocked(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn
    from conference_pipeline.production import notebooklm_stage

    ctx = _podcast_context(_podcast_workspace(tmp_path), "nb-canary")
    _seed_podcast_predecessors(ctx)
    engine = pn.FakeNotebookLMEngine()
    outcome = notebooklm_stage.run_notebooklm_podcast_stage(
        ctx, deck_ids=["01_alpha_topic"], ack_live_notebooklm=True, engine=engine, transcriber=_passing_transcriber()
    )
    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["blockers"][0]["code"] == "notebooklm_podcast_deck_scope_filtered"
    assert outcome["receipt"]["blockers"][0]["full_scope_deck_count"] == 4
    assert (ctx.workspace / "audio" / "01_alpha_topic" / "notebooklm_podcast" / "podcast.m4a").is_file()
    assert engine.generate_calls == 1
    assert outcome["receipt"]["stage_contract"] is None


def test_notebooklm_fake_engine_end_to_end_satisfies_runtime_contract_and_resumes(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn
    from conference_pipeline import stage_contracts
    from conference_pipeline.production import notebooklm_stage

    summaries: list[dict] = []

    def summary_writer(receipt: dict) -> dict:
        summaries.append(receipt)
        return {"stage": receipt["stage_id"], "status": receipt["status"], "via": "runner-hook"}

    ctx = _podcast_context(_podcast_workspace(tmp_path), "nb-full", summary_writer=summary_writer)
    _seed_podcast_predecessors(ctx)
    outcome = notebooklm_stage.run_notebooklm_podcast_stage(
        ctx, ack_live_notebooklm=True, engine=pn.FakeNotebookLMEngine(), transcriber=_passing_transcriber()
    )
    receipt = outcome["receipt"]
    assert outcome["status"] == "succeeded", receipt["blockers"]
    checks = receipt["checks"]
    assert checks["notebooklm_podcast_aggregate_sha256"]
    assert checks["storage_state_present"] is True and checks["engine_version"] == "fake"
    assert receipt["provider_calls_made"] is True
    assert receipt["stage_contract"]["status"] == "pass", receipt["stage_contract"]["blockers"]
    token = receipt["owner_execution_evidence"]["podcast.notebooklm_audio_overview"]
    assert token.endswith(checks["notebooklm_podcast_aggregate_sha256"])
    assert outcome["summary"]["via"] == "runner-hook" and summaries[-1]["stage_id"] == "notebooklm-podcast"
    independent = stage_contracts.verify_stage_contract(
        workspace=ctx.workspace, run_id=ctx.run_id, stage_id="notebooklm-podcast", receipt=receipt, mode="live",
        strict_execution=True,
    ).to_dict()
    assert independent["status"] == "pass", independent["blockers"]

    resumed = notebooklm_stage.run_notebooklm_podcast_stage(
        ctx, ack_live_notebooklm=True, engine=pn.FakeNotebookLMEngine(preflight_ok=False),
        transcriber=_passing_transcriber(),
    )
    assert resumed["status"] == "succeeded" and resumed["summary"].get("resumed") is True


def test_notebooklm_quota_halt_is_blocked_and_retryable(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn
    from conference_pipeline.production import notebooklm_stage

    ctx = _podcast_context(_podcast_workspace(tmp_path), "nb-quota")
    _seed_podcast_predecessors(ctx)
    outcome = notebooklm_stage.run_notebooklm_podcast_stage(
        ctx, ack_live_notebooklm=True, engine=pn.FakeNotebookLMEngine(fail_quota_on_call=2),
        transcriber=_passing_transcriber(),
    )
    assert outcome["status"] == "blocked"
    blocker = outcome["receipt"]["blockers"][0]
    assert blocker["code"] == "notebooklm_quota_exhausted" and blocker["retryable"] is True
    assert outcome["receipt"]["checks"]["deck_success_count"] == 1
    auth = notebooklm_stage.run_notebooklm_podcast_stage(
        ctx, ack_live_notebooklm=True, engine=pn.FakeNotebookLMEngine(preflight_ok=False),
        transcriber=_passing_transcriber(),
    )
    assert auth["receipt"]["blockers"][0]["code"] == "notebooklm_auth_preflight_failed"
    assert auth["receipt"]["checks"]["storage_state_present"] is False


def test_icml_runner_dispatches_notebooklm_podcast_with_its_flags() -> None:
    runner = _load_icml_runner()
    stage = runner.STAGES["notebooklm-podcast"]
    assert (stage.order, stage.receipt_filename, stage.implemented) == (10, "stage-10-notebooklm-podcast.json", True)
    args = runner.parse_args(
        ["--stage", "notebooklm-podcast", "--deck-ids", "01_a, 02_b", "--ack-live-notebooklm", "--notebooklm-length", "short"]
    )
    assert args.ack_live_notebooklm is True and args.notebooklm_length == "short" and args.deck_ids == "01_a, 02_b"
    assert runner.orchestrator_spec().handlers["notebooklm-podcast"] is runner.run_notebooklm_podcast


# ---------------------------------------------------------------------------
# Stage 10 presentation shell: the zero-spend gates
# ---------------------------------------------------------------------------

PRESENTATION_STAGES = {
    **PODCAST_STAGES,
    "podcast": _Stage("podcast", 9, "stage-09-podcast.json", True, "rerun podcast"),
}


def _presentation_context(workspace: Path, run_id: str, **overrides: object) -> ProductionRunnerContext:
    return _context(workspace, run_id, stages=PRESENTATION_STAGES, **overrides)


def _seed_presentation_inputs(ctx: ProductionRunnerContext) -> None:
    from tests.lineage_fixtures import stamp_upstream_stage_receipts

    run_dir = ctx.run_dir()
    (run_dir / "backup").mkdir(parents=True, exist_ok=True)
    for stage_id in ("qa-gap-fill-synthesis", "podcast"):
        ctx.receipt_path(stage_id).write_text(
            json.dumps({"stage_id": stage_id, "status": "succeeded", "mode": "live", "stage_contract": {"status": "pass"}}),
            encoding="utf-8",
        )
    (run_dir / "backup" / "podcast-backup-write-receipt.json").write_text(
        json.dumps({"status": "succeeded"}), encoding="utf-8"
    )
    (ctx.workspace / "analysis" / "executive_summary.md").write_text("# Executive summary\n", encoding="utf-8")
    (ctx.workspace / "analysis" / "trend_report.md").write_text("# Trends\n", encoding="utf-8")
    stamp_upstream_stage_receipts(run_dir)


class _NeverRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str]):
        self.calls.append(command)
        raise AssertionError(f"no subprocess may run behind a gate: {command[:2]}")


def test_presentation_blocks_on_missing_inputs_without_running_anything(tmp_path: Path) -> None:
    from conference_pipeline.production import presentation_stage

    ctx = _presentation_context(_workspace(tmp_path, "test2026"), "pres-missing")
    runner = _NeverRunner()
    outcome = presentation_stage.run_presentation_stage(ctx, resume=False, command_runner=runner)
    assert outcome["status"] == "blocked"
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["presentation_required_inputs_blocked"]
    codes = {item["code"] for item in outcome["receipt"]["checks"]["input_blockers"]}
    assert "presentation_stage8_receipt_missing_or_malformed" in codes
    assert "presentation_backup_receipt_missing" in codes
    assert outcome["receipt"]["receipt_type"] == "conference_pipeline.testrunner_production_stage_receipt"
    assert outcome["receipt"]["provider_calls_made"] is False
    assert runner.calls == []


def test_presentation_blocks_without_provider_cost_approval(tmp_path: Path) -> None:
    from conference_pipeline.production import presentation_stage

    ctx = _presentation_context(_workspace(tmp_path, "test2026"), "pres-cost")
    _seed_presentation_inputs(ctx)
    outcome = presentation_stage.run_presentation_stage(
        ctx, resume=False, provider_cost_approved=False, command_runner=_NeverRunner()
    )
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["presentation_provider_cost_not_approved"]
    assert outcome["receipt"]["checks"]["provider_cost_approval"]["approved"] is False


def test_presentation_refuses_an_unknown_canary_deck(tmp_path: Path) -> None:
    from conference_pipeline.production import presentation_stage

    ctx = _presentation_context(_workspace(tmp_path, "test2026"), "pres-unknown")
    _seed_presentation_inputs(ctx)
    outcome = presentation_stage.run_presentation_stage(
        ctx, resume=False, provider_cost_approved=True, deck_ids=["99_nope"], command_runner=_NeverRunner()
    )
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["presentation_deck_scope_filter_unknown_deck"]
    checks = outcome["receipt"]["checks"]
    assert checks["unknown_deck_ids"] == ["99_nope"]
    assert set(checks["known_deck_ids"]) == {"01_alpha_topic", "executive_summary_short", "executive_summary_long"}


def test_presentation_cost_approval_comes_from_the_runner_hook(tmp_path: Path) -> None:
    from conference_pipeline.production import presentation_stage

    seen: list[bool] = []

    def cost_approval(flag: bool) -> dict:
        seen.append(flag)
        return {"approved": True, "approval_sources": ["env:TEST_APPROVED"], "required_before_paid_provider_calls": True}

    ctx = _presentation_context(_workspace(tmp_path, "test2026"), "pres-hook", cost_approval=cost_approval)
    _seed_presentation_inputs(ctx)
    outcome = presentation_stage.run_presentation_stage(
        ctx, resume=False, provider_cost_approved=False, deck_ids=["99_nope"], command_runner=_NeverRunner()
    )
    assert seen == [False]
    # The hook approved, so the run got past the cost gate to the deck-scope gate.
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["presentation_deck_scope_filter_unknown_deck"]


def test_icml_runner_presentation_flags_reach_the_shared_stage() -> None:
    runner = _load_icml_runner()
    args = runner.parse_args(
        [
            "--stage", "presentation", "--deck-ids", "01_a", "--figure-route", "compiler",
            "--presentation-substage", "deck_figures", "--accept-visual-qc-findings",
            "--accept-presentation-review-findings", "01_a,02_b", "--adopt-existing-outputs",
            "--blueprint-approved-by", "op", "--visual-qc-max-rounds", "1",
        ]
    )
    assert args.figure_route == "compiler" and args.presentation_substage == "deck_figures"
    assert args.accept_visual_qc_findings is True and args.adopt_existing_outputs is True
    assert args.accept_presentation_review_findings == "01_a,02_b" and args.visual_qc_max_rounds == 1
    default = runner.parse_args(["--stage", "presentation"])
    assert default.figure_route == "compiler"
