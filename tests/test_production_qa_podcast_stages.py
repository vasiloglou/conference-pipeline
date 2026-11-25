"""Shared Stage 8 (QA/synthesis) and Stage 9 (podcast) modules.

The two stages used to live, near-duplicated, in every conference runner.
These tests pin what the shared modules do for ANY runner, with no hooks:
a minimal new conference (``minimal_new_conference`` fixture, plus a second
synthetic slug) runs both stages end to end against a fake synthesis client,
fake owner commands and a fake provider-receipt summary, under the default
strict receipt convention and its own identity; resume never re-spends; the
gates block before any spend; the default podcast layout and title come from
the workspace's conference identity, never from a literal.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conference_pipeline.analysis.narrative import DEFAULT_DEEP_NARRATIVE_MODEL
from conference_pipeline.model_governance import default_model, default_provider
from conference_pipeline.production import ProductionRunnerContext
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production import podcast_stage, qa_synthesis_stage, stage_table
from conference_pipeline.production.context import UPSTREAM_LINEAGE_NOT_FRESH
from conference_pipeline.production.receipts import StageReceiptConvention, StrictStageConvention
from conference_pipeline.provider_receipts import active_blockers, scan_unblock_receipts
from conference_pipeline.workflow_runs import validate_run_id, write_json_atomic
from tests.lineage_fixtures import stamp_upstream_stage_receipts
from tests.orchestrator_golden import materialize_fixture_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_NOW = "2032-01-01T00:00:00Z"

# A new conference declares only the stages it dispatches; Stage 8 reads the
# Stage 7 receipt, so the table declares analysis without running it.
STAGES = stage_table.from_registry(
    next_actions={
        "analysis": "Not run by this orchestrator yet.",
        "qa-gap-fill-synthesis": "Repair the Stage 8 inputs, then rerun qa-gap-fill-synthesis.",
        "podcast": "Repair the Stage 9 inputs, then rerun podcast.",
    },
    not_implemented={"analysis"},
)


def _context(workspace: Path, run_id: str) -> ProductionRunnerContext:
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
def workspace(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    if request.param == "minimal_new_conference":
        return materialize_fixture_workspace("minimal_new_conference", tmp_path)
    return _second_workspace(tmp_path)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def seed_stage7(ctx: ProductionRunnerContext, *, topic_count: int = 3) -> dict[str, Any]:
    """Stage 7 outputs and receipts that pass every Stage 8 input gate."""

    workspace = ctx.workspace
    topics: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for index in range(1, topic_count + 1):
        topic_id = f"{index:02d}_fixture_topic"
        markdown = workspace / "analysis" / f"{topic_id}.md"
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(f"# Fixture Topic {index}\n\nBody for {topic_id}.\n", encoding="utf-8")
        for rel in (f"provenance/{topic_id}.md", f"provenance/{topic_id}.json", f"slide_briefs/{topic_id}.md"):
            (workspace / "analysis" / rel).parent.mkdir(parents=True, exist_ok=True)
            (workspace / "analysis" / rel).write_text("{}\n", encoding="utf-8")
        paths = {
            "markdown_path": f"analysis/{topic_id}.md",
            "provenance_path": f"analysis/provenance/{topic_id}.md",
            "provenance_json_path": f"analysis/provenance/{topic_id}.json",
            "slide_brief_path": f"analysis/slide_briefs/{topic_id}.md",
        }
        topics.append(
            {
                "id": topic_id,
                "title": f"Fixture Topic {index}",
                **paths,
                "markdown_sha256": hashlib.sha256(markdown.read_bytes()).hexdigest(),
                "deep_validation": {"status": "pass"},
                "narrative_model": DEFAULT_DEEP_NARRATIVE_MODEL,
            }
        )
        citations.append(
            {
                "topic_id": topic_id,
                **paths,
                "extraction_record_ids": [
                    f"fixture-extraction-{topic_id}-{n}" for n in range(qa_synthesis_stage.STAGE8_EXTRACTION_ID_FLOOR)
                ],
                "vector_store_chunk_ids": [f"fixture-chunk-{topic_id}"],
                "research_ids": [f"fixture-research-{topic_id}"],
            }
        )
        _write_json(
            workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
            {
                "topic_id": topic_id,
                "topic_reconciliation": {
                    "status": "succeeded",
                    "reference_validation_status": "succeeded",
                    "reference_validation_counts": {"citation_count": 1, "ledger_entry_count": 1},
                },
            },
        )
    content_sha = hashlib.sha256(json.dumps(topics, sort_keys=True).encode("utf-8")).hexdigest()
    _write_json(workspace / ctx.analysis_manifest, {"content_sha256": content_sha, "topics": topics})
    _write_json(workspace / "analysis" / "analysis_citation_manifest.json", {"topics": citations})
    run_dir = ctx.run_dir()
    _write_json(
        ctx.receipt_path("analysis"),
        {
            "status": "succeeded",
            "stage_contract": {"status": "pass"},
            "checks": {"analysis_manifest_content_sha256": content_sha},
        },
    )
    counts = ("invalid_reference_count", "unknown_reference_count", "title_mismatch_count", "bare_reference_warning_count")
    _write_json(
        run_dir / qa_synthesis_stage.TOPIC_RECONCILIATION_RECEIPT, {"status": "succeeded", "topic_count": topic_count}
    )
    _write_json(
        run_dir / qa_synthesis_stage.TOPIC_REFERENCE_VALIDATION_RECEIPT,
        {"status": "succeeded", "checks": {key: 0 for key in counts}},
    )
    _write_json(
        run_dir / qa_synthesis_stage.TOPIC_REFERENCE_WORKFLOW_RECEIPT,
        {
            "status": "succeeded",
            "checks": {"final_validation_status": "succeeded", **{f"final_{key}": 0 for key in counts}},
        },
    )
    stamp_upstream_stage_receipts(run_dir)
    return {"topics": topics, "content_sha": content_sha}


class FakeSynthesisClient:
    """Governed provider/model, canned long text; never touches a network."""

    def __init__(self) -> None:
        task = qa_synthesis_stage.SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID
        self.provider = default_provider(task)
        self.model = default_model(task)
        self.calls: list[str] = []

    def synthesize(self, *, kind: str, system_prompt: str, user_prompt: str) -> Any:
        assert "Citation pool" in user_prompt
        self.calls.append(kind)
        return SimpleNamespace(text=f"Fixture {kind} synthesis sentence. " * 200, input_tokens=100, output_tokens=50)


class Factory:
    """``client_factory`` that records every construction."""

    def __init__(self) -> None:
        self.clients: list[FakeSynthesisClient] = []

    def __call__(self, ctx: ProductionRunnerContext) -> FakeSynthesisClient:
        client = FakeSynthesisClient()
        self.clients.append(client)
        return client


def must_not_construct(ctx: ProductionRunnerContext) -> Any:
    raise AssertionError("paid synthesis client constructed where it must not be")


class FakeCommands:
    """Stand-in for the script-generation and audio-render owner commands."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        if podcast_stage.SCRIPT_GENERATOR in command:
            segment = " ".join(["A source-backed fixture segment."] * 40)
            _write_json(output_dir / "script.json", {"title": "Fixture", "segments": [{"text": segment}] * 2})
            (output_dir / "script.md").write_text("# Fixture\n", encoding="utf-8")
            (output_dir / "show_notes.md").write_text("# Notes\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "script ok", "")
        if podcast_stage.AUDIO_RENDERER in command:
            (output_dir / "podcast.mp3").write_bytes(b"fixture-mp3")
            _write_json(output_dir / "audio_manifest.json", {"segment_count": 2, "audio_file_count": 2})
            return subprocess.CompletedProcess(command, 0, "audio ok", "")
        raise AssertionError(command)


def must_not_run(command: list[str]) -> Any:
    raise AssertionError(f"owner command ran where it must not: {command}")


def fake_receipt_summary(workspace: Path, run_id: str | None = None, stage_id: str | None = None) -> dict[str, Any]:
    return {"canonical_receipt_files_scanned": 2, "canonical_total_estimated_cost_usd": 0.02}


def run_stage8(ctx: ProductionRunnerContext, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("provider_cost_approved", True)
    return qa_synthesis_stage.run_qa_gap_fill_synthesis_stage(ctx, **kwargs)


def run_stage9(ctx: ProductionRunnerContext, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("provider_cost_approved", True)
    kwargs.setdefault("receipt_summary", fake_receipt_summary)
    return podcast_stage.run_podcast_stage(ctx, **kwargs)


# ---------------------------------------------------------------------------
# A new conference with no hooks runs Stage 8 and Stage 9 end to end
# ---------------------------------------------------------------------------


def test_new_conference_runs_stage8_and_stage9_with_no_hooks(workspace: Path) -> None:
    ctx = _context(workspace, f"{workspace.name}-w4")
    seed_stage7(ctx)
    factory = Factory()

    stage8 = run_stage8(ctx, client_factory=factory)

    assert stage8["status"] == "succeeded", stage8["receipt"]["blockers"]
    receipt8 = stage8["receipt"]
    assert receipt8["receipt_type"] == ctx.receipt_type
    assert receipt8["lineage"]["code_identity"]["runner"] == ctx.runner_id
    assert receipt8["stage_contract"]["status"] == "pass"
    assert [client.calls for client in factory.clients] == [["executive_summary", "trend_report"]]
    assert set(receipt8["owner_execution_evidence"]) == {
        qa_synthesis_stage.OWNER_COVERAGE_GAP_FILL,
        qa_synthesis_stage.OWNER_SYNTHESIS_TREND_REPORT,
    }
    assert (workspace / "analysis" / "qa" / "coverage_gap_report.json").is_file()
    assert "## Source Anchors" in (workspace / ctx.analysis_executive_summary).read_text(encoding="utf-8")
    summary = json.loads((ctx.run_dir() / "summary.json").read_text(encoding="utf-8"))
    assert [item["stage_id"] for item in summary["stages"]] == ["qa-gap-fill-synthesis"]

    commands = FakeCommands()
    stage9 = run_stage9(ctx, command_runner=commands)

    assert stage9["status"] == "succeeded", stage9["receipt"]["blockers"]
    receipt9 = stage9["receipt"]
    checks = receipt9["checks"]
    name = ctx.conference_display_name()
    overview = f"audio/{ctx.slug}_overview/focused_podcast"
    assert checks["podcast_mp3_path"] == f"{overview}/podcast.mp3"
    assert commands.commands[0][commands.commands[0].index("--title") + 1] == f"{name}: What Changed in AI Research"
    package = json.loads((workspace / overview / "audio_ready_package.json").read_text(encoding="utf-8"))
    assert package["receipt_type"] == f"conference_pipeline.{ctx.slug}_podcast_audio_ready_package"
    assert (workspace / "analysis" / "podcast_source.md").read_text(encoding="utf-8").startswith(f"# {name} Podcast Source")
    assert set(receipt9["owner_execution_evidence"]) == {
        podcast_stage.OWNER_FOCUSED_SCRIPT,
        podcast_stage.OWNER_RENDER_AUDIO,
    }
    summary = json.loads((ctx.run_dir() / "summary.json").read_text(encoding="utf-8"))
    assert [item["stage_id"] for item in summary["stages"]] == ["qa-gap-fill-synthesis", "podcast"]

    # Resume accepts both receipts as-is: no client, no command, same bytes.
    before8 = ctx.receipt_path("qa-gap-fill-synthesis").read_bytes()
    before9 = ctx.receipt_path("podcast").read_bytes()
    assert run_stage8(ctx, client_factory=must_not_construct)["status"] == "succeeded"
    assert run_stage9(ctx, command_runner=must_not_run)["status"] == "succeeded"
    assert ctx.receipt_path("qa-gap-fill-synthesis").read_bytes() == before8
    assert ctx.receipt_path("podcast").read_bytes() == before9


def test_minimal_orchestrator_dispatches_stage8_with_empty_hooks(tmp_path: Path) -> None:
    """The shared orchestrator, ``ConferenceHooks()`` and run_shared_stage are enough."""

    workspace = materialize_fixture_workspace("minimal_new_conference", tmp_path)
    run_id = "newconf2032-w4-cli"

    def run_dir(ws: Path, rid: str) -> Path:
        return ws / "storage" / "workflow-runs" / validate_run_id(rid)

    conventions = production_orchestrator.RunConventions(
        build_receipt=lambda **kwargs: {"stage_id": kwargs["stage"].id, "status": kwargs["status"]},
        write_receipt=lambda ws, rid, stage, receipt: write_json_atomic(run_dir(ws, rid) / stage.receipt_filename, receipt),
        write_summary=lambda **kwargs: production_orchestrator.write_run_summary(
            **kwargs,
            summary_path=run_dir(kwargs["workspace"], kwargs["run_id"]) / "summary.json",
            generated_at=FROZEN_NOW,
            unblock_receipts=scan_unblock_receipts(kwargs["workspace"]),
            write_json=write_json_atomic,
        ),
        active_blockers=active_blockers,
        context=_context,
        run_dir=run_dir,
        now=lambda: FROZEN_NOW,
        validate_run_id=validate_run_id,
    )
    spec = production_orchestrator.OrchestratorSpec(
        display_name="New Conference",
        description="Run New Conference production stages.",
        stages=STAGES,
        handlers={
            "qa-gap-fill-synthesis": lambda args, stage: production_orchestrator.run_shared_stage(
                conventions,
                args,
                stage,
                lambda ctx: qa_synthesis_stage.run_qa_gap_fill_synthesis_stage(
                    ctx, resume=bool(args.resume), provider_cost_approved=True, client_factory=Factory()
                ),
            ),
            "podcast": lambda args, stage: production_orchestrator.run_shared_stage(
                conventions,
                args,
                stage,
                lambda ctx: podcast_stage.run_podcast_stage(
                    ctx,
                    resume=bool(args.resume),
                    provider_cost_approved=True,
                    command_runner=FakeCommands(),
                    receipt_summary=fake_receipt_summary,
                ),
            ),
        },
    )
    assert spec.hooks == production_orchestrator.NO_HOOKS
    assert production_orchestrator.spec_problems(spec) == []
    seed_stage7(_context(workspace.resolve(), run_id))

    argv = ["--workspace", str(workspace), "--run-id", run_id]
    assert production_orchestrator.run([*argv, "--stage", "qa-gap-fill-synthesis"], spec) == 0
    assert production_orchestrator.run([*argv, "--stage", "podcast"], spec) == 0
    summary = json.loads((run_dir(workspace, run_id) / "summary.json").read_text(encoding="utf-8"))
    assert (summary["status"], summary["stages"][0]["stage_id"]) == ("succeeded", "podcast")


# ---------------------------------------------------------------------------
# Gates block before any spend
# ---------------------------------------------------------------------------


def test_stage8_blocks_on_a_missing_stage7_receipt_without_constructing_a_client(workspace: Path) -> None:
    ctx = _context(workspace, "w4-missing-stage7")
    seed_stage7(ctx)
    ctx.receipt_path("analysis").unlink()

    outcome = run_stage8(ctx, client_factory=must_not_construct)

    assert outcome["status"] == "blocked"
    checks = outcome["receipt"]["checks"]
    assert checks["blocker_code"] == "stage8_required_inputs_blocked"
    assert checks["provider_calls_made"] is False
    assert {item["code"] for item in checks["input_blockers"]} == {"stage8_input_missing"}


def test_stage8_without_cost_approval_blocks_before_constructing_a_client(workspace: Path) -> None:
    ctx = _context(workspace, "w4-unapproved")
    seed_stage7(ctx)

    outcome = run_stage8(ctx, provider_cost_approved=False, client_factory=must_not_construct)

    assert outcome["receipt"]["checks"]["blocker_code"] == "stage8_provider_cost_not_approved"
    # The deterministic ($0) QA reports were still written and bound.
    assert outcome["receipt"]["checks"]["citation_manifest_sha256"]


def test_stage8_an_injected_client_is_the_callers_own_approval(workspace: Path) -> None:
    ctx = _context(workspace, "w4-injected")
    seed_stage7(ctx)
    client = FakeSynthesisClient()

    outcome = run_stage8(ctx, provider_cost_approved=False, client=client, client_factory=must_not_construct)

    assert outcome["status"] == "succeeded"
    assert client.calls == ["executive_summary", "trend_report"]


def test_stage8_blocks_uncited_synthesis(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(workspace, "w4-uncited")
    seed_stage7(ctx)
    monkeypatch.setattr(qa_synthesis_stage, "append_stage8_source_anchor", lambda text, **_kwargs: text)

    outcome = run_stage8(ctx, client_factory=Factory())

    checks = outcome["receipt"]["checks"]
    assert checks["blocker_code"] == "stage8_synthesis_validation_blocked"
    assert "stage8_synthesis_digest_binding_missing" in {item["code"] for item in checks["synthesis_blockers"]}


def test_podcast_blocks_before_spend_when_a_stage8_output_was_edited(workspace: Path) -> None:
    ctx = _context(workspace, "w4-edited")
    seed_stage7(ctx)
    assert run_stage8(ctx, client_factory=Factory())["status"] == "succeeded"
    summary_path = ctx.workspace / ctx.analysis_executive_summary
    summary_path.write_text(summary_path.read_text(encoding="utf-8") + "\nOut-of-band edit.\n", encoding="utf-8")

    outcome = run_stage9(ctx, command_runner=must_not_run)

    checks = outcome["receipt"]["checks"]
    assert (outcome["status"], checks["blocker_code"]) == ("blocked", UPSTREAM_LINEAGE_NOT_FRESH)
    [blocker] = checks["upstream_lineage_blockers"]
    assert (blocker["upstream_stage_id"], blocker["paths"]) == ("qa-gap-fill-synthesis", ["analysis/executive_summary.md"])


def test_podcast_without_cost_approval_blocks_before_any_command(workspace: Path) -> None:
    ctx = _context(workspace, "w4-podcast-unapproved")
    seed_stage7(ctx)
    assert run_stage8(ctx, client_factory=Factory())["status"] == "succeeded"

    outcome = run_stage9(ctx, provider_cost_approved=False, command_runner=must_not_run)

    assert outcome["receipt"]["checks"]["blocker_code"] == "podcast_provider_cost_not_approved"


def test_podcast_script_only_outcome_cannot_succeed(workspace: Path) -> None:
    ctx = _context(workspace, "w4-script-only")
    seed_stage7(ctx)
    assert run_stage8(ctx, client_factory=Factory())["status"] == "succeeded"
    fake = FakeCommands()

    def no_audio(command: list[str]) -> subprocess.CompletedProcess[str]:
        if podcast_stage.AUDIO_RENDERER in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return fake(command)

    outcome = run_stage9(ctx, command_runner=no_audio)

    assert outcome["receipt"]["checks"]["blocker_code"] == "podcast_audio_outputs_missing"


@pytest.mark.parametrize("require", [True, False])
def test_stage8_output_status_gate_is_a_layout_setting(tmp_path: Path, require: bool) -> None:
    ctx = _context(_second_workspace(tmp_path), "w4-status-gate")
    inputs = podcast_stage.required_input_paths(ctx)
    layout = podcast_stage.PodcastLayout.for_context(ctx)
    layout = podcast_stage.PodcastLayout(**{**layout.__dict__, "require_stage8_output_status": require})
    receipt = {"status": "succeeded", "stage_contract": {"status": "pass"}, "checks": {"trend_report_status": "pass"}}

    blockers = podcast_stage._input_blockers(
        ctx,
        inputs,
        layout=layout,
        stage8_receipt=receipt,
        stage8_error=None,
        analysis_manifest={},
        analysis_manifest_error=None,
    )

    codes = {item["code"] for item in blockers}
    assert ("podcast_stage8_executive_summary_status_not_pass" in codes) is require


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


def test_both_receipt_conventions_satisfy_the_protocol(tmp_path: Path) -> None:
    ctx = _context(_second_workspace(tmp_path), "w4-protocol")
    strict = StrictStageConvention(ctx, qa_synthesis_stage.STAGE_ID)
    summary_convention = production_orchestrator.RunnerSummaryConvention(
        conventions=None,  # type: ignore[arg-type]
        stage=ctx.stage("podcast"),
        workspace=ctx.workspace,
        run_id=ctx.run_id,
        mode="live",
        started_at=FROZEN_NOW,
        block_stage=lambda **_kwargs: (1, {}),
        upstream_gate=lambda **_kwargs: None,
        verify_contract=lambda **_kwargs: None,
        load_json=lambda path: (None, "unused"),
    )
    assert isinstance(strict, StageReceiptConvention)
    assert isinstance(summary_convention, StageReceiptConvention)
    assert strict.start() is None


def test_summary_convention_records_gate_time_input_digests(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def upstream_gate(**kwargs: Any) -> None:
        seen.update(kwargs["checks"])

    convention = production_orchestrator.RunnerSummaryConvention(
        conventions=None,  # type: ignore[arg-type]
        stage=STAGES["qa-gap-fill-synthesis"],
        workspace=tmp_path,
        run_id="w4-digests",
        mode="live",
        started_at=FROZEN_NOW,
        block_stage=lambda **_kwargs: (1, {}),
        upstream_gate=upstream_gate,
        verify_contract=lambda **_kwargs: None,
        load_json=lambda path: (None, "unused"),
    )
    assert convention.lineage_gate({"env_file": {}}, input_digests={"stage7_receipt_sha256": "a" * 64}) is None
    assert seen == {"env_file": {}, "stage7_receipt_sha256": "a" * 64}


def test_shared_modules_carry_no_conference_literal() -> None:
    for module in (qa_synthesis_stage, podcast_stage):
        text = Path(module.__file__).read_text(encoding="utf-8").lower()
        for token in ("icml", "cvpr", "iclr", "neurips"):
            assert token not in text, (module.__name__, token)
