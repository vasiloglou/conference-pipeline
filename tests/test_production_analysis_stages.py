"""Shared Stage 7A (exhaustive extraction) and Stage 7 (analysis) modules.

Both stages used to live, near-duplicated, in every conference runner. These
tests pin what the shared modules do for ANY runner, with no hooks: a
minimal new conference (``minimal_new_conference`` fixture, plus a second
synthetic slug) runs both stages end to end against a fake extraction owner
and a fake deep-narrative client, under the default strict receipt
convention and its own identity; resume never re-runs; every gate blocks
before any spend; the source-pack policy flags only add what they name.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conference_pipeline.analysis.narrative import (
    DEFAULT_DEEP_NARRATIVE_MODEL,
    deterministic_narrative_topic_report,
)
from conference_pipeline.indexing.corpus_vector_store import workspace_relpath
from conference_pipeline.production import (
    ProductionRunnerContext,
    analysis_stage,
    exhaustive_extraction_stage,
    stage_table,
)
from tests.lineage_fixtures import stamp_upstream_stage_receipts
from tests.orchestrator_golden import materialize_fixture_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "newconf-analysis-fixture"
TOPIC_WORDS = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "sigma"
)
# The registry's analysis contract still pins twelve topics (see the batch-R note in the W5 report).
TOPIC_IDS = tuple(f"{index:02d}_{word}_topic" for index, word in enumerate(TOPIC_WORDS, start=1))

STAGES = stage_table.from_registry(
    next_actions={
        "corpus-inventory-index": "Not run by this orchestrator yet.",
        "exhaustive-topic-extraction": "Repair the Stage 7A inputs, then rerun exhaustive-topic-extraction.",
        "analysis": "Repair the Stage 7 inputs, then rerun analysis.",
    },
    not_implemented={"corpus-inventory-index"},
)


def _context(workspace: Path) -> ProductionRunnerContext:
    slug = workspace.name
    return ProductionRunnerContext(
        workspace=workspace,
        run_id=RUN_ID,
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
        workspace = materialize_fixture_workspace("minimal_new_conference", tmp_path)
    else:
        workspace = _second_workspace(tmp_path)
    return _context(workspace)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def seed_stage6(ctx: ProductionRunnerContext) -> None:
    """A succeeded Stage 6 receipt over a tiny papers/artifacts chunk store."""

    workspace = ctx.workspace
    vector_root = workspace / "storage" / "vector-store" / RUN_ID
    rows = {
        "papers": [
            {
                "chunk_id": f"paper-chunk-{topic}-{index}",
                "artifact_id": f"paper_text:{topic}-{index}",
                "source_path": f"https://newconf.example.test/papers/{topic}-{index}",
                "text": f"Evidence sentence {index} about {topic.split('_')[1]} research methods.",
            }
            for topic in TOPIC_IDS
            for index in range(4)
        ],
        "artifacts": [
            {
                "chunk_id": f"artifact-chunk-{topic}",
                "artifact_id": f"speech_transcript:{topic}",
                "source_path": f"https://newconf.example.test/talks/{topic}",
                "text": f"Talk transcript about {topic.split('_')[1]} research methods.",
            }
            for topic in TOPIC_IDS
        ],
    }
    stores = {}
    for name, chunk_rows in rows.items():
        chunks_path = vector_root / "stores" / name / "chunks.jsonl"
        _write_jsonl(chunks_path, chunk_rows)
        stores[name] = {"chunks_path": workspace_relpath(workspace, chunks_path)}
    index_digest_path = _write_json(vector_root / "index_digest.json", {"stores": stores})
    counts = {name: len(chunk_rows) for name, chunk_rows in rows.items()}
    _write_json(
        ctx.receipt_path("corpus-inventory-index"),
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "stage_id": "corpus-inventory-index",
            "mode": "live",
            "status": "succeeded",
            "checks": {
                "vector_embeddings_written": True,
                "chunk_counts": counts,
                "embedded_chunk_counts": counts,
                "index_digest_sha256": sha256_of(index_digest_path),
                "output_inventory": {"sha256": "a" * 64},
            },
        },
    )
    stamp_upstream_stage_receipts(ctx.run_dir())


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_extraction_owner(calls: list[dict[str, Any]], *, status: str = "succeeded"):
    def owner(workspace_root: Path, *, run_id: str, force_rebuild: bool = False) -> dict[str, Any]:
        calls.append({"run_id": run_id, "force_rebuild": force_rebuild})
        root = Path(workspace_root) / "storage" / "extractions" / run_id
        for topic in TOPIC_IDS:
            _write_json(root / "by-topic" / f"{topic}.json", {"topic_id": topic, "artifacts": []})
        _write_json(root / "by-artifact" / "fixture-artifact.json", {"artifact_id": "fixture-artifact"})
        _write_json(root / "v15_extraction_coverage.json", {"coverage_ratio": 1.0})
        return {"schema_version": 1, "run_id": run_id, "status": status, "blockers": []}

    return owner


def _must_not_run(label: str):
    def run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{label} must not run")

    return run


# --------------------------------------------------------------------------
# Stage 7A
# --------------------------------------------------------------------------


def run_7a(ctx: ProductionRunnerContext, owner: Any, *, resume: bool = False) -> dict[str, Any]:
    return exhaustive_extraction_stage.run_exhaustive_topic_extraction_stage(
        ctx, resume=resume, env_file=None, owner=owner
    )


def test_exhaustive_extraction_runs_with_no_hooks_under_its_own_identity(ctx: ProductionRunnerContext) -> None:
    seed_stage6(ctx)
    calls: list[dict[str, Any]] = []

    outcome = run_7a(ctx, fake_extraction_owner(calls))

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    assert calls == [{"run_id": RUN_ID, "force_rebuild": True}]
    assert receipt["receipt_type"] == ctx.receipt_type
    assert receipt["lineage"]["code_identity"]["runner"] == ctx.runner_id
    assert receipt["stage_contract"]["status"] == "pass"
    assert receipt["provider_calls_made"] == 0
    evidence = receipt["owner_execution_evidence"]
    assert evidence and all(item["owner_module"] == exhaustive_extraction_stage.OWNER_MODULE for item in evidence.values())
    assert ctx.slug not in exhaustive_extraction_stage.OWNER_MODULE
    assert json.loads((ctx.run_dir() / "summary.json").read_text())["status"] == "succeeded"


def test_exhaustive_extraction_resume_rewrites_the_verified_receipt_without_rerunning(
    ctx: ProductionRunnerContext,
) -> None:
    seed_stage6(ctx)
    assert run_7a(ctx, fake_extraction_owner([]))["status"] == "succeeded"
    first = json.loads(ctx.receipt_path("exhaustive-topic-extraction").read_text())

    resumed = run_7a(ctx, _must_not_run("extraction owner"), resume=True)

    assert resumed["status"] == "succeeded"
    assert resumed["receipt"]["lineage"] == first["lineage"]
    assert resumed["receipt"]["output_digest"] == first["output_digest"]


def test_exhaustive_extraction_blocks_without_stage6_and_never_runs_the_owner(ctx: ProductionRunnerContext) -> None:
    outcome = run_7a(ctx, _must_not_run("extraction owner"))

    assert outcome["status"] == "blocked"
    [blocker] = outcome["receipt"]["blockers"]
    assert blocker["code"] == exhaustive_extraction_stage.STAGE6_REQUIRED
    assert blocker["affected_stage"] == "exhaustive-topic-extraction"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False


def test_exhaustive_extraction_records_an_owner_failure_as_a_blocked_receipt(ctx: ProductionRunnerContext) -> None:
    seed_stage6(ctx)

    def raising_owner(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("fixture owner failure")

    raised = run_7a(ctx, raising_owner)
    not_succeeded = run_7a(ctx, fake_extraction_owner([], status="blocked"))

    assert [b["code"] for b in raised["receipt"]["blockers"]] == [exhaustive_extraction_stage.OWNER_RAISED]
    assert raised["receipt"]["blockers"][0]["error"] == "fixture owner failure"
    assert [b["code"] for b in not_succeeded["receipt"]["blockers"]] == [exhaustive_extraction_stage.OWNER_NOT_SUCCEEDED]


# --------------------------------------------------------------------------
# Stage 7
# --------------------------------------------------------------------------


def seed_analysis_inputs(ctx: ProductionRunnerContext, *, topic_ids: tuple[str, ...] = TOPIC_IDS) -> None:
    """Stage 6, Stage 7A and every post-full-read chain receipt the analysis gates read."""

    seed_stage6(ctx)
    assert run_7a(ctx, fake_extraction_owner([]))["status"] == "succeeded"
    workspace, target = ctx.workspace, ctx.run_dir()
    final_topics = [
        {"topic_id": topic_id, "title": f"{topic_id.split('_')[1].title()} research methods"} for topic_id in topic_ids
    ]
    _write_json(workspace / "analysis" / "topic_strategy.json", {"final_topics": final_topics, "topic_count": len(final_topics)})
    _write_json(
        analysis_stage.topic_strategy_receipt_path(workspace, RUN_ID),
        {
            "status": "succeeded",
            "mode": "live",
            "provider_receipt_paths": ["storage/provider-calls/topic-strategy/fixture.json"],
            "final_topics": final_topics,
        },
    )
    extractions = workspace / "storage" / "extractions" / RUN_ID
    for index, topic_id in enumerate(topic_ids, start=1):
        _write_json(
            extractions / "llm-full-read" / "by-topic" / f"{topic_id}.json",
            {
                "unit_count": 1,
                "items": [
                    {
                        "unit_id": f"unit-{index}",
                        "input_kind": "paper_abstract",
                        "confidence": 0.9,
                        "important": True,
                        "summary": f"Full-read summary for {topic_id}.",
                        "reason": "fixture",
                        "evidence_phrases": ["fixture phrase"],
                        "source_artifact_path": f"storage/corpus/paper-text/{topic_id}.json",
                        "metadata": {"title": f"Paper {index}", "paper_id": f"paper-{index}"},
                    }
                ],
            },
        )
        _write_json(extractions / "llm-full-read" / "by-unit" / f"unit-{index}.json", {"unit_id": f"unit-{index}"})
    count = len(topic_ids)
    _write_json(
        target / analysis_stage.LLM_FULL_READ_RECEIPT,
        {
            "status": "succeeded",
            "scope": "all",
            "limit": None,
            "checks": {
                "completion_state": "complete",
                "expected_unit_count": count,
                "completed_unit_count": count,
                "failed_count": 0,
            },
        },
    )
    _write_json(target / analysis_stage.TOPIC_RECONCILIATION_RECEIPT, {"status": "succeeded", "source_packs_enriched": True})
    validation_counts = {
        "ledger_entry_count": count,
        "citation_count": count,
        "invalid_reference_count": 0,
        "unknown_reference_count": 0,
        "title_mismatch_count": 0,
        "bare_reference_warning_count": 0,
    }
    _write_json(
        target / analysis_stage.TOPIC_REFERENCE_VALIDATION_RECEIPT,
        {"status": "succeeded", "topic_count": count, "checks": {"completed_topic_count": count, **validation_counts}},
    )
    _write_json(
        target / analysis_stage.RECONCILIATION_REFERENCE_RECEIPT,
        {
            "status": "succeeded",
            "checks": {
                "final_validation_status": "succeeded",
                "final_topic_count": count,
                "final_completed_topic_count": count,
                **{f"final_{key}": value for key, value in validation_counts.items()},
            },
        },
    )
    for name in (analysis_stage.DEEP_DIVE_QUESTION_SELECTION_RECEIPT, analysis_stage.DUAL_DEEP_DIVE_RESEARCH_RECEIPT):
        _write_json(target / name, {"status": "succeeded", "provider_calls_made": True})
    for topic_id in topic_ids:
        reconciliation = workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md"
        reconciliation.parent.mkdir(parents=True, exist_ok=True)
        reconciliation.write_text(f"# {topic_id}\n\nFixture reconciliation.\n", encoding="utf-8")
        _write_json(workspace / "analysis" / "topic_reference_ledgers" / f"{topic_id}.json", {"topic_id": topic_id, "entries": ["e"]})
        _write_json(
            workspace / "analysis" / "topic_reference_validation" / f"{topic_id}.json",
            {"topic_id": topic_id, "status": "succeeded"},
        )
    stamp_upstream_stage_receipts(target)


class FakeNarrativeClient:
    """The paid deep-narrative client's stand-in: the provider-free report for the prompted pack."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.model = DEFAULT_DEEP_NARRATIVE_MODEL
        self.request_count = 0

    def synthesize(self, *, kind: str, system_prompt: str, user_prompt: str) -> Any:
        del kind, system_prompt
        self.request_count += 1
        for path in sorted((self.workspace / "analysis" / "narrative_source_packs").glob("*.json")):
            pack = json.loads(path.read_text(encoding="utf-8"))
            if str(pack.get("research_id") or "") in user_prompt:
                return SimpleNamespace(text=deterministic_narrative_topic_report(pack))
        raise AssertionError("prompt names no source pack")


def enrich_packs_like_reconciliation(ctx: ProductionRunnerContext) -> None:
    """An earlier zero-spend pass built the packs and 07D enriched them."""

    for path in sorted((ctx.workspace / "analysis" / "narrative_source_packs").glob("*.json")):
        pack = json.loads(path.read_text(encoding="utf-8"))
        pack["topic_reconciliation"] = {
            "status": "succeeded",
            "model": "fixture-reconciliation",
            "reference_validation_status": "succeeded",
            "reference_validation_counts": {
                "citation_count": 1,
                "ledger_entry_count": 1,
                "invalid_reference_count": 0,
                "unknown_reference_count": 0,
                "title_mismatch_count": 0,
                "bare_reference_warning_count": 0,
            },
        }
        _write_json(path, pack)


def run_7(
    ctx: ProductionRunnerContext,
    *,
    approved: bool = True,
    resume: bool = False,
    client: Any | None = None,
    owners: analysis_stage.AnalysisOwners | None = None,
    settings: analysis_stage.AnalysisSettings = analysis_stage.AnalysisSettings(),
) -> dict[str, Any]:
    if owners is None:
        owners = analysis_stage.AnalysisOwners(narrative_client=lambda: client)
    return analysis_stage.run_analysis_stage(
        ctx, resume=resume, provider_cost_approved=approved, env_file=None, settings=settings, owners=owners
    )


def test_analysis_builds_packs_then_succeeds_with_no_hooks(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)
    settings = analysis_stage.AnalysisSettings(expected_topic_count=len(TOPIC_IDS))
    # First pass: zero-spend packs (07D would enrich them next).
    analysis_stage.build_analysis_source_packs(
        workspace=ctx.workspace,
        run_id=RUN_ID,
        stage6_receipt=json.loads(ctx.receipt_path("corpus-inventory-index").read_text()),
        index_digest=json.loads((ctx.workspace / "storage/vector-store" / RUN_ID / "index_digest.json").read_text()),
        topic_specs=analysis_stage.load_analysis_topic_specs(ctx.workspace, run_id=RUN_ID, expected_topic_count=len(TOPIC_IDS)),
    )
    enrich_packs_like_reconciliation(ctx)
    client = FakeNarrativeClient(ctx.workspace)

    outcome = run_7(ctx, client=client, settings=settings)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    assert client.request_count == len(TOPIC_IDS)
    receipt = outcome["receipt"]
    assert receipt["receipt_type"] == ctx.receipt_type
    assert receipt["stage_contract"]["status"] == "pass"
    assert receipt["checks"]["topic_strategy_topic_ids"] == list(TOPIC_IDS)
    manifest = json.loads((ctx.workspace / "analysis" / "analysis_manifest.json").read_text())
    assert manifest["generator"] == f"conference_pipeline.{ctx.slug}_production.analysis@v1"
    evidence = receipt["owner_execution_evidence"]
    assert evidence and not any(value.endswith(":evidence_unavailable") for value in evidence.values())


def test_analysis_blocks_before_any_pack_or_client_without_full_read(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)
    (ctx.run_dir() / analysis_stage.LLM_FULL_READ_RECEIPT).unlink()
    owners = analysis_stage.AnalysisOwners(
        build_source_packs=_must_not_run("source packs"),
        narrative_client=_must_not_run("narrative client"),
        render_narratives=_must_not_run("renderer"),
    )

    outcome = run_7(ctx, owners=owners, settings=analysis_stage.AnalysisSettings(expected_topic_count=len(TOPIC_IDS)))

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["blocker_code"] == "analysis_llm_full_read_extraction_required"
    assert outcome["receipt"]["provider_calls_made"] is False


def test_analysis_builds_packs_but_blocks_before_spend_on_a_missing_chain_receipt(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)
    (ctx.run_dir() / analysis_stage.DUAL_DEEP_DIVE_RESEARCH_RECEIPT).unlink()
    owners = analysis_stage.AnalysisOwners(
        narrative_client=_must_not_run("narrative client"), render_narratives=_must_not_run("renderer")
    )

    outcome = run_7(ctx, owners=owners, settings=analysis_stage.AnalysisSettings(expected_topic_count=len(TOPIC_IDS)))

    assert outcome["receipt"]["checks"]["blocker_code"] == "analysis_dual_deep_dive_research_required"
    assert outcome["receipt"]["checks"]["source_packs"]["source_pack_count"] == len(TOPIC_IDS)


# --------------------------------------------------------------------------
# Source-pack policy
# --------------------------------------------------------------------------


def _build_packs(ctx: ProductionRunnerContext, policy: analysis_stage.SourcePackPolicy, **kwargs: Any) -> dict[str, Any]:
    return analysis_stage.build_analysis_source_packs(
        workspace=ctx.workspace,
        run_id=RUN_ID,
        stage6_receipt=json.loads(ctx.receipt_path("corpus-inventory-index").read_text()),
        index_digest=json.loads((ctx.workspace / "storage/vector-store" / RUN_ID / "index_digest.json").read_text()),
        topic_specs=analysis_stage.load_analysis_topic_specs(ctx.workspace, run_id=RUN_ID, expected_topic_count=len(TOPIC_IDS)),
        policy=policy,
        **kwargs,
    )


def test_default_policy_records_only_the_shared_fields(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)

    result = _build_packs(ctx, analysis_stage.DEFAULT_SOURCE_PACK_POLICY)

    assert set(result) == {"source_pack_count", "source_pack_paths", "query_plan_paths", "insufficient_topics"}
    plan = json.loads((ctx.workspace / result["query_plan_paths"][0]).read_text())
    assert "source_families" not in plan and "evidence_importance" not in plan
    pack = json.loads((ctx.workspace / result["source_pack_paths"][0]).read_text())
    assert set(pack["chunks"][0]) == {"chunk_id", "extraction_record_id", "source_artifact_path", "quote"}
    assert "selected_importance_counts" not in pack["llm_full_read_evidence"]


def test_policy_flags_add_what_they_name_and_scope_carries_packs_forward(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)
    policy = analysis_stage.SourcePackPolicy(
        compact_chunk=analysis_stage.compact_chunk_with_metadata,
        matched_paper_priority=True,
        evidence_importance=True,
        record_source_families=True,
        scoped_rebuilds=True,
    )
    _build_packs(ctx, policy)
    carried = (ctx.workspace / "analysis/narrative_source_packs" / f"{TOPIC_IDS[1]}.json").read_bytes()

    result = _build_packs(ctx, policy, topic_scope=frozenset({TOPIC_IDS[0]}))

    assert result["topic_scope"] == [TOPIC_IDS[0]]
    assert result["carried_forward_topics"] == list(TOPIC_IDS[1:])
    assert (ctx.workspace / "analysis/narrative_source_packs" / f"{TOPIC_IDS[1]}.json").read_bytes() == carried
    plan = json.loads((ctx.workspace / result["query_plan_paths"][0]).read_text())
    assert "source_families" in plan and "evidence_importance" in plan
    assert "priority_paper_count" in plan["paper_matching"]
    pack = json.loads((ctx.workspace / result["source_pack_paths"][0]).read_text())
    assert analysis_stage.COMPACT_ABSTRACT_WINDOW_KEY in pack["chunks"][0]
    assert "selected_importance_counts" in pack["llm_full_read_evidence"]


def test_identical_selections_are_refused_only_when_the_policy_says_so(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)
    real = analysis_stage.select_topic_source_chunks

    def identical_selector(**kwargs: Any) -> Any:
        selection = real(**kwargs)
        return SimpleNamespace(
            selected=selection.selected,
            fallback=selection.fallback,
            family_counts=selection.family_counts,
            identical_selection_groups=[list(TOPIC_IDS[:2])],
        )

    _build_packs(ctx, analysis_stage.SourcePackPolicy(), selector=identical_selector)
    with pytest.raises(analysis_stage.IdenticalSourcePacksError) as raised:
        _build_packs(ctx, analysis_stage.SourcePackPolicy(reject_identical_selections=True), selector=identical_selector)
    assert raised.value.groups == [list(TOPIC_IDS[:2])]
    assert str(raised.value) == "analysis_source_packs_identical:" + ",".join(TOPIC_IDS[:2])


# --------------------------------------------------------------------------
# W5 Phase B: resume never re-pays; no renderer without approval
# --------------------------------------------------------------------------


def succeed_analysis(ctx: ProductionRunnerContext) -> FakeNarrativeClient:
    seed_analysis_inputs(ctx)
    _build_packs(ctx, analysis_stage.DEFAULT_SOURCE_PACK_POLICY)
    enrich_packs_like_reconciliation(ctx)
    client = FakeNarrativeClient(ctx.workspace)
    assert run_7(ctx, client=client)["status"] == "succeeded"
    return client


def test_analysis_resume_returns_a_verified_receipt_without_any_work(ctx: ProductionRunnerContext) -> None:
    succeed_analysis(ctx)
    before = ctx.receipt_path("analysis").read_bytes()
    owners = analysis_stage.AnalysisOwners(
        build_source_packs=_must_not_run("source packs"),
        narrative_client=_must_not_run("narrative client"),
        render_narratives=_must_not_run("renderer"),
    )

    resumed = run_7(ctx, resume=True, owners=owners)

    assert resumed["status"] == "succeeded"
    assert ctx.receipt_path("analysis").read_bytes() == before


def test_analysis_resume_reuses_current_narratives_without_a_paid_call(ctx: ProductionRunnerContext) -> None:
    succeed_analysis(ctx)
    reports = {path.name: path.read_bytes() for path in (ctx.workspace / "analysis").glob("[0-9][0-9]_*.md")}
    ctx.receipt_path("analysis").unlink()  # no receipt to short-circuit on: the gates re-run
    client = FakeNarrativeClient(ctx.workspace)

    approved = run_7(ctx, resume=True, client=client)
    ctx.receipt_path("analysis").unlink()
    unapproved = run_7(ctx, resume=True, approved=False, owners=analysis_stage.AnalysisOwners(
        narrative_client=_must_not_run("narrative client")
    ))

    for outcome in (approved, unapproved):
        assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
        reuse = outcome["receipt"]["checks"]["narrative_reuse"]
        assert reuse["reused_topic_ids"] == list(TOPIC_IDS) and reuse["pending_topic_ids"] == []
        assert outcome["receipt"]["provider_calls_made"] is False
    assert client.request_count == 0
    assert {path.name: path.read_bytes() for path in (ctx.workspace / "analysis").glob("[0-9][0-9]_*.md")} == reports


def test_analysis_resume_resynthesizes_only_topics_whose_inputs_changed(ctx: ProductionRunnerContext) -> None:
    succeed_analysis(ctx)
    ctx.receipt_path("analysis").unlink()
    changed = TOPIC_IDS[0]
    provenance_path = ctx.workspace / "analysis" / "provenance" / f"{changed}.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["narrative_model"] = "an-older-narrative-model"
    _write_json(provenance_path, provenance)
    report_before = (ctx.workspace / "analysis" / f"{changed}.md").read_bytes()

    blocked = run_7(
        ctx,
        resume=True,
        approved=False,
        owners=analysis_stage.AnalysisOwners(
            narrative_client=_must_not_run("narrative client"), render_narratives=_must_not_run("renderer")
        ),
    )
    assert blocked["receipt"]["checks"]["blocker_code"] == "analysis_provider_cost_approval_required"
    assert blocked["receipt"]["checks"]["narrative_reuse"]["pending_topic_ids"] == [changed]
    assert (ctx.workspace / "analysis" / f"{changed}.md").read_bytes() == report_before

    client = FakeNarrativeClient(ctx.workspace)
    resumed = run_7(ctx, resume=True, client=client)
    assert resumed["status"] == "succeeded", resumed["receipt"]["blockers"]
    assert client.request_count == 1


def test_analysis_without_approval_blocks_before_the_renderer(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)
    owners = analysis_stage.AnalysisOwners(
        narrative_client=_must_not_run("narrative client"), render_narratives=_must_not_run("renderer")
    )

    outcome = run_7(ctx, approved=False, owners=owners)

    assert outcome["receipt"]["checks"]["blocker_code"] == "analysis_provider_cost_approval_required"
    assert "narrative_status" not in outcome["receipt"]["checks"]
    assert not list((ctx.workspace / "analysis").glob("[0-9][0-9]_*.md"))


# --------------------------------------------------------------------------
# W5 Phase B: the topic count comes from the conference's topic strategy
# --------------------------------------------------------------------------


def _rewrite_strategy(ctx: ProductionRunnerContext, *, keep: int, declared: Any = "unset") -> None:
    strategy_path = ctx.workspace / "analysis" / "topic_strategy.json"
    strategy = json.loads(strategy_path.read_text())
    strategy["final_topics"] = strategy["final_topics"][:keep]
    if declared == "unset":
        strategy.pop("topic_count", None)
    else:
        strategy["topic_count"] = declared
    _write_json(strategy_path, strategy)
    receipt_path = analysis_stage.topic_strategy_receipt_path(ctx.workspace, RUN_ID)
    receipt = json.loads(receipt_path.read_text())
    receipt["final_topics"] = strategy["final_topics"]
    _write_json(receipt_path, receipt)


def test_topic_specs_follow_the_strategy_count_by_default(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)
    for keep, declared in ((12, 12), (5, 5), (5, "unset")):
        _rewrite_strategy(ctx, keep=keep, declared=declared)
        specs = analysis_stage.load_analysis_topic_specs(ctx.workspace, run_id=RUN_ID)
        assert [spec["id"] for spec in specs] == list(TOPIC_IDS[:keep])


@pytest.mark.parametrize(
    ("keep", "declared", "code"),
    [
        (5, 12, "analysis_topic_strategy_topic_count_mismatch"),
        (5, True, "analysis_topic_strategy_topic_count_mismatch"),
        (0, "unset", "analysis_topic_strategy_final_topics_missing"),
    ],
)
def test_topic_specs_refuse_an_inconsistent_strategy(ctx: ProductionRunnerContext, keep: int, declared: Any, code: str) -> None:
    seed_analysis_inputs(ctx)
    _rewrite_strategy(ctx, keep=keep, declared=declared)

    with pytest.raises(RuntimeError, match=f"^{code}$"):
        analysis_stage.load_analysis_topic_specs(ctx.workspace, run_id=RUN_ID)
    outcome = run_7(ctx, owners=analysis_stage.AnalysisOwners(build_source_packs=_must_not_run("source packs")))
    assert outcome["receipt"]["checks"]["blocker_code"] == code


def test_a_pinned_topic_count_still_blocks_on_any_other_count(ctx: ProductionRunnerContext) -> None:
    seed_analysis_inputs(ctx)
    _rewrite_strategy(ctx, keep=5, declared=5)

    with pytest.raises(RuntimeError, match="^analysis_topic_strategy_requires_12_final_topics$"):
        analysis_stage.load_analysis_topic_specs(ctx.workspace, run_id=RUN_ID, expected_topic_count=12)


def test_identical_selections_block_the_stage_with_a_receipt(ctx: ProductionRunnerContext) -> None:
    """W5 Phase B: the refusal becomes a blocked receipt, never an escaping exception."""

    seed_analysis_inputs(ctx)

    def identical(**_kwargs: Any) -> Any:
        raise analysis_stage.IdenticalSourcePacksError([list(TOPIC_IDS[:2])])

    outcome = run_7(
        ctx,
        owners=analysis_stage.AnalysisOwners(
            build_source_packs=identical, narrative_client=_must_not_run("narrative client")
        ),
    )

    assert outcome["status"] == "blocked"
    assert [b["code"] for b in outcome["receipt"]["blockers"]] == ["analysis_source_packs_identical"]
    assert outcome["receipt"]["checks"]["identical_source_pack_groups"] == [list(TOPIC_IDS[:2])]
