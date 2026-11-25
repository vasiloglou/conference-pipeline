"""Shared Stage 6 (corpus-inventory-index) module.

Stage 6 used to live in every conference runner. These tests pin what the
shared module does for ANY runner: a minimal new conference (the
``minimal_new_conference`` fixture, plus a second synthetic slug) indexes with
no hooks in both receipt formats on the incremental store with the local
deterministic ``fake`` embedding provider, a resume re-embeds nothing and a
new artifact embeds only its own chunks. Declared record builders, speech
hooks and gates are exercised with fakes. Nothing touches the network or a
provider, and no real workspace is opened.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conference_pipeline.production import ProductionRunnerContext, corpus_index_stage, stage_table
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production.download_corpus_stage import LedgerConventions
from conference_pipeline.production.hooks import ConferenceHooks, validate_hooks
from conference_pipeline.provider_receipts import utc_now
from conference_pipeline.receipt_lineage import sha256_file
from conference_pipeline.transcription_ocr import TRANSCRIPTION_OCR_INVENTORY
from conference_pipeline.workflow_runs import validate_run_id
from tests.orchestrator_golden import materialize_fixture_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "conference_pipeline" / "production" / "corpus_index_stage.py"
RUN_ID = "newconf-stage6"
STAGES = stage_table.from_registry(
    next_actions={
        "download-corpus": "Repair the download-corpus blockers, then rerun download-corpus.",
        "transcription-ocr": "Repair the transcription-ocr blockers, then rerun transcription-ocr.",
        "corpus-inventory-index": "Repair the corpus-inventory-index blockers, then rerun corpus-inventory-index.",
    },
)
PAPER_TEXT = (
    "Sparse mixture models for long-context retrieval. We study how routing "
    "sparsity changes recall on long documents and report ablations. "
) * 12
SECOND_PAPER_TEXT = (
    "Graph neural operators for weather forecasting. A mesh-based encoder "
    "processes reanalysis fields and we evaluate skill at ten-day lead times. "
) * 12


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
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
def ctx(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProductionRunnerContext:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    if request.param == "minimal_new_conference":
        workspace = materialize_fixture_workspace("minimal_new_conference", tmp_path / "conferences")
    else:
        workspace = _second_workspace(tmp_path)
    return _context(workspace.resolve())


def run_conventions(ctx: ProductionRunnerContext, written_blockers: list[dict[str, Any]]) -> production_orchestrator.RunConventions:
    def build_receipt(*, workspace: Path, run_id: str, stage: Any, mode: str, status: str, started_at: str,
                      checks: dict[str, Any], error: str | None = None,
                      blocker_receipt_path: str | None = None) -> dict[str, Any]:
        receipt = {
            "schema_version": 1,
            "receipt_type": ctx.receipt_type,
            "run_id": run_id,
            "stage_id": stage.id,
            "stage_order": stage.order,
            "mode": mode,
            "status": status,
            "started_at": started_at,
            "checks": checks,
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


# --------------------------------------------------------------------------
# Lane-inventory format (the default): Stage 4/5 handoff of the same run
# --------------------------------------------------------------------------


def seed_lane_handoff(ctx: ProductionRunnerContext, papers: dict[str, str], speech: list[dict[str, Any]] | None = None) -> None:
    workspace = ctx.workspace
    download_inventory = _write_json(
        workspace / corpus_index_stage.DOWNLOAD_CORPUS_INVENTORY,
        {"stage_id": "download-corpus", "status": "succeeded", "local_artifacts": [], "direct_download_artifacts": []},
    )
    download_sha = sha256_file(download_inventory)
    _write_json(
        ctx.receipt_path("download-corpus"),
        {"stage_id": "download-corpus", "status": "succeeded", "checks": {"download_corpus_inventory_sha256": download_sha}},
    )
    paper_records = []
    for name, text in papers.items():
        source = workspace / "papers" / f"{name}.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            source.write_bytes(f"%PDF-1.4\n% synthetic {name}\n".encode("ascii"))
        output = workspace / "storage" / "transcription-ocr" / "text" / f"{name}.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        paper_records.append(
            {
                "source_path": f"papers/{name}.pdf",
                "status": "succeeded",
                "output_path": f"storage/transcription-ocr/text/{name}.txt",
                "title": f"Paper {name}",
            }
        )
    transcription_inventory = _write_json(
        workspace / TRANSCRIPTION_OCR_INVENTORY,
        {
            "lanes": {
                "paper_text_extraction": {"records": paper_records},
                "session_speech_transcription": {"records": speech or []},
            }
        },
    )
    _write_json(
        ctx.receipt_path("transcription-ocr"),
        {
            "stage_id": "transcription-ocr",
            "status": "succeeded",
            "checks": {
                "download_corpus_inventory_sha256": download_sha,
                "output_inventory": {"sha256": sha256_file(transcription_inventory)},
            },
        },
    )


LANE_SETTINGS = corpus_index_stage.LaneInventorySettings(
    inventory_receipt_type="conference_pipeline.synthetic_corpus_inventory",
    paper_curation_min_text_bytes=0,
)


def run_lane(ctx: ProductionRunnerContext, blockers: list[dict[str, Any]], **overrides: Any) -> tuple[int, dict[str, Any]]:
    options: dict[str, Any] = {
        "workspace": ctx.workspace,
        "run_id": ctx.run_id,
        "stage": ctx.stage("corpus-inventory-index"),
        "download_stage": ctx.stage("download-corpus"),
        "transcription_stage": ctx.stage("transcription-ocr"),
        "mode": "live",
        "env_file_status": lambda: {"attempted": False, "exists": False, "loaded": False},
        "full_rebuild": False,
        "settings": LANE_SETTINGS,
    }
    options.update(overrides)
    return corpus_index_stage.run_lane_inventory_stage(run_conventions(ctx, blockers), **options)


def _stage6_receipt(ctx: ProductionRunnerContext) -> dict[str, Any]:
    return _read_json(ctx.receipt_path("corpus-inventory-index"))


def test_missing_download_receipt_blocks_before_anything_is_indexed(ctx: ProductionRunnerContext) -> None:
    blockers: list[dict[str, Any]] = []
    exit_code, summary = run_lane(ctx, blockers)

    assert exit_code == 1 and summary["status"] == "blocked"
    receipt = _stage6_receipt(ctx)
    assert receipt["receipt_type"] == ctx.receipt_type
    assert receipt["checks"]["blocker_code"] == "download_corpus_stage_not_succeeded"
    assert receipt["checks"]["download_corpus_stage_receipt_present"] is False
    assert receipt["checks"]["provider_cost_posture"]["paid_provider_calls_made"] is False
    assert blockers[0]["remediation_path"] == "rerun download-corpus before corpus-inventory-index"
    assert not (ctx.workspace / corpus_index_stage.CORPUS_INVENTORY).exists()


def test_stage5_inventory_digest_drift_blocks(ctx: ProductionRunnerContext) -> None:
    seed_lane_handoff(ctx, {"p-001": PAPER_TEXT})
    inventory_path = ctx.workspace / TRANSCRIPTION_OCR_INVENTORY
    inventory_path.write_text(inventory_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    exit_code, _summary = run_lane(ctx, [])

    assert exit_code == 1
    assert _stage6_receipt(ctx)["checks"]["blocker_code"] == "transcription_ocr_inventory_digest_mismatch"


def test_no_hooks_indexes_incrementally_with_fake_embeddings(ctx: ProductionRunnerContext) -> None:
    seed_lane_handoff(ctx, {"p-001": PAPER_TEXT})

    exit_code, summary = run_lane(ctx, [])

    assert exit_code == 0, summary
    receipt = _stage6_receipt(ctx)
    checks = receipt["checks"]
    assert receipt["status"] == "succeeded"
    assert checks["artifact_counts"]["by_source_family"] == {"paper_text": 1}
    assert checks["chunk_counts"]["papers"] > 0
    assert checks["embedded_chunk_counts"] == checks["chunk_counts"]
    inventory = _read_json(ctx.workspace / corpus_index_stage.CORPUS_INVENTORY)
    assert inventory["receipt_type"] == "conference_pipeline.synthetic_corpus_inventory"
    assert inventory["artifact_records"][0]["metadata"]["title"] == "Paper p-001"
    store = ctx.workspace / "storage" / "vector-store" / RUN_ID / "stores" / "papers"
    assert (store / "chunk_manifest.json").is_file()
    assert (store / "keyword_index.json").is_file()
    first_inventory_bytes = (ctx.workspace / corpus_index_stage.CORPUS_INVENTORY).read_bytes()

    # Resume over an unchanged corpus: the equivalent inventory is preserved
    # byte-for-byte and nothing is re-embedded.
    exit_code, _summary = run_lane(ctx, [])
    assert exit_code == 0
    assert (ctx.workspace / corpus_index_stage.CORPUS_INVENTORY).read_bytes() == first_inventory_bytes
    update = _stage6_receipt(ctx)["checks"]["vector_materialization"]["embedding_update"]
    assert all(store_update["chunks_embedded_now"] == 0 for store_update in update.values())

    # A new paper embeds only its own chunks.
    seed_lane_handoff(ctx, {"p-001": PAPER_TEXT, "p-002": SECOND_PAPER_TEXT})
    exit_code, _summary = run_lane(ctx, [])
    assert exit_code == 0
    checks = _stage6_receipt(ctx)["checks"]
    update = checks["vector_materialization"]["embedding_update"]["papers"]
    assert checks["artifact_counts"]["by_source_family"] == {"paper_text": 2}
    assert 0 < update["chunks_embedded_now"] < checks["chunk_counts"]["papers"]


def test_speech_hooks_restore_and_bind_and_their_blockers_name_the_stage_reason(ctx: ProductionRunnerContext) -> None:
    transcript = ctx.workspace / "storage" / "transcription-ocr" / "speech" / "talk-1.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("hello world " * 40, encoding="utf-8")
    audio = ctx.workspace / "cache" / "talk-1.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio")
    speech = [
        {
            "source_path": "cache/talk-1.mp3",
            "status": "succeeded",
            "transcript_path": "storage/transcription-ocr/speech/talk-1.txt",
            "recording_id": "rec-1",
            "source_title": "Opening keynote",
        }
    ]
    seed_lane_handoff(ctx, {"p-001": PAPER_TEXT}, speech=speech)
    calls: list[str] = []

    def restore(_state: Any, record: corpus_index_stage.SpeechRecord) -> bool:
        calls.append("restore")
        record.lane_record = {**record.lane_record, "talk_title": "Restored"}
        return True

    def failing_bind(workspace: Path, video_id: str, transcript_path: Path) -> tuple[None, str]:
        calls.append(f"bind:{video_id}")
        return None, "talk_map_stale_for_transcript"

    builders = (
        corpus_index_stage.paper_text_lanes(),
        corpus_index_stage.speech_transcript_lanes(
            hooks=(
                corpus_index_stage.SpeechTranscriptHook("restore", restore=restore),
                corpus_index_stage.talk_map_speech_hook(failing_bind, video_id_field="recording_id"),
            )
        ),
    )
    blockers: list[dict[str, Any]] = []
    exit_code, _summary = run_lane(ctx, blockers, hooks=ConferenceHooks(record_builders=builders))

    assert exit_code == 1
    assert calls == ["restore", "bind:rec-1"]
    checks = _stage6_receipt(ctx)["checks"]
    assert checks["blocker_code"] == "talk_map_chain_invalid"
    assert checks["blockers"][0]["talk_map_blocker"] == "talk_map_stale_for_transcript"
    assert "split_event_recordings.py" in blockers[0]["remediation_path"]


def test_curate_summaries_and_a_gate_failure(ctx: ProductionRunnerContext) -> None:
    seed_lane_handoff(ctx, {"p-001": PAPER_TEXT})

    def curate(assembly: corpus_index_stage.IndexAssembly) -> None:
        assembly.summaries["titles"] = {"titled": len(assembly.artifact_records)}

    def gate(assembly: corpus_index_stage.IndexAssembly) -> None:
        assembly.gate_failure = corpus_index_stage.GateFailure(
            blocker_code="scope_not_proven", reason="scope not proven", remediation="prove it", checks={"scope": 1}
        )

    builders = (
        *corpus_index_stage.default_lane_inventory_builders(),
        corpus_index_stage.RecordBuilder("titles", curate=curate),
        corpus_index_stage.RecordBuilder("scope", gate=gate),
    )
    exit_code, _summary = run_lane(ctx, [], hooks=ConferenceHooks(record_builders=builders))

    assert exit_code == 1
    checks = _stage6_receipt(ctx)["checks"]
    assert checks["blocker_code"] == "scope_not_proven" and checks["scope"] == 1


def test_untitled_indexable_transcript_blocks(ctx: ProductionRunnerContext) -> None:
    transcript = ctx.workspace / "storage" / "transcription-ocr" / "speech" / "talk-1.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("hello world " * 40, encoding="utf-8")
    (ctx.workspace / "cache").mkdir(parents=True, exist_ok=True)
    (ctx.workspace / "cache" / "talk-1.mp3").write_bytes(b"audio")
    seed_lane_handoff(
        ctx,
        {"p-001": PAPER_TEXT},
        speech=[{"source_path": "cache/talk-1.mp3", "status": "succeeded",
                 "transcript_path": "storage/transcription-ocr/speech/talk-1.txt"}],
    )

    exit_code, _summary = run_lane(ctx, [])

    assert exit_code == 1
    checks = _stage6_receipt(ctx)["checks"]
    assert checks["blocker_code"] == "speech_transcript_title_missing"
    assert checks["untitled_transcript_count"] == 1


# --------------------------------------------------------------------------
# Admitted-sources format: digest-bound program-ledger handoff, no hooks
# --------------------------------------------------------------------------


def _extractable_pdf(path: Path, text: str) -> bytes:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(36, 36, 560, 800), text)
    data = document.tobytes(garbage=4, deflate=True, no_new_id=True)
    document.close()
    path.write_bytes(data)
    return data


CORPUS_FREEZE = Path("storage/corpus-freeze/corpus_freeze.json")
SOURCE_DISCOVERY = Path("storage/source-discovery/source_discovery_inventory.json")


def seed_program_ledger(ctx: ProductionRunnerContext, papers: dict[str, str]) -> None:
    import hashlib

    workspace = ctx.workspace
    artifacts = []
    for key, text in papers.items():
        relative = f"storage/download-corpus/papers/{key}.pdf"
        data = _extractable_pdf(workspace / relative, text)
        artifacts.append(
            {"artifact_kind": "paper_pdf", "local_path": relative, "sha256": hashlib.sha256(data).hexdigest(), "request_key": key}
        )
    _write_json(
        workspace / corpus_index_stage.DOWNLOAD_CORPUS_INVENTORY,
        {
            "stage_id": "download-corpus",
            "status": "succeeded",
            "manifest_sha256": "ledger-digest-1",
            "artifacts": artifacts,
            "lineage": {"phase_02_inventory_sha256": "inventory-digest-1"},
        },
    )
    _write_json(
        ctx.receipt_path("download-corpus"),
        {
            "stage_id": "download-corpus",
            "stage_order": STAGES["download-corpus"].order,
            "mode": "live",
            "status": "succeeded",
            "blockers": [],
            "checks": {"download_corpus_ledger_sha256": "ledger-digest-1"},
        },
    )
    _write_json(
        ctx.receipt_path("transcription-ocr"),
        {"stage_id": "transcription-ocr", "stage_order": STAGES["transcription-ocr"].order, "mode": "live",
         "status": "succeeded", "blockers": []},
    )
    _write_json(workspace / CORPUS_FREEZE, {"stage_id": "corpus-freeze", "status": "succeeded"})
    _write_json(
        workspace / SOURCE_DISCOVERY,
        {"inventory_sha256": "inventory-digest-1", "program_items": [], "source_records": []},
    )


def ledger_conventions(ctx: ProductionRunnerContext, contract_status: str = "pass") -> LedgerConventions:
    return LedgerConventions(
        stages=STAGES,
        run_dir=lambda workspace, run_id: workspace / "storage" / "workflow-runs" / validate_run_id(run_id),
        now=utc_now,
        write_json=lambda path, payload: _write_json(path, payload),
        write_bytes=lambda path, payload: path.write_bytes(payload),
        finalize_receipt=lambda receipt, **kwargs: ctx.finalize(receipt, stage_id=kwargs["stage_id"]),
        verify_contract=lambda **_kwargs: SimpleNamespace(to_dict=lambda: {"status": contract_status}),
        read_json_object=corpus_index_stage.read_json_object,
        receipt_type=ctx.receipt_type,
        owner_id="acquisition.corpus_inventory_index.synthetic",
        stage_id="corpus-inventory-index",
    )


ADMITTED_SETTINGS = corpus_index_stage.AdmittedSourcesSettings(
    inventory_receipt_type="conference_pipeline.synthetic_corpus_inventory",
    owner_function="run_corpus_inventory_index_stage",
    corpus_freeze_path=CORPUS_FREEZE,
    source_discovery_path=SOURCE_DISCOVERY,
    paper_curation_min_text_bytes=0,
)


def run_admitted(ctx: ProductionRunnerContext, *, resume: bool = True, contract_status: str = "pass",
                 hooks: ConferenceHooks | None = None) -> dict[str, Any]:
    return corpus_index_stage.run_admitted_sources_stage(
        ledger_conventions(ctx, contract_status),
        workspace=ctx.workspace,
        run_id=ctx.run_id,
        resume=resume,
        full_rebuild=False,
        verify_chunk_cache=True,
        settings=ADMITTED_SETTINGS,
        **({"hooks": hooks} if hooks is not None else {}),
    )


def test_admitted_sources_without_a_ledger_blocks_with_every_admission_blocker(ctx: ProductionRunnerContext) -> None:
    outcome = run_admitted(ctx)

    assert outcome["status"] == "blocked"
    codes = [blocker["code"] for blocker in outcome["receipt"]["blockers"]]
    assert codes[:2] == ["download_corpus_ledger_missing", "download_corpus_stage_not_succeeded"]
    assert "transcription_ocr_stage_not_succeeded" in codes
    assert codes[-1] == "source_discovery_inventory_missing"
    assert outcome["corpus_inventory"] is None


def test_admitted_sources_no_hooks_indexes_main_papers_incrementally(ctx: ProductionRunnerContext) -> None:
    seed_program_ledger(ctx, {"n-001": PAPER_TEXT})

    outcome = run_admitted(ctx, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    assert receipt["required_script_ids"] == ["acquisition.corpus_inventory_index.synthetic"]
    assert receipt["owner_execution_evidence"]["acquisition.corpus_inventory_index.synthetic"]["owner_function"] == (
        "run_corpus_inventory_index_stage"
    )
    checks = receipt["checks"]
    assert checks["artifact_counts"]["by_source_family"] == {"paper_text": 1}
    assert checks["download_corpus_stage_receipt_resolved_run_id"] is None
    assert checks["source_discovery_inventory_status"] == "ok"
    record = outcome["corpus_inventory"]["artifact_records"][0]
    assert record["artifact_id"] == "paper:n-001" and record["indexable_text"] is True
    assert record["storage_lane"] == "storage/corpus/paper-text"

    resumed = run_admitted(ctx, resume=True)
    assert resumed["status"] == "succeeded"
    assert all(update["chunks_embedded_now"] == 0 for update in resumed["receipt"]["checks"]["embedding_update"].values())
    # The first succeeded receipt is preserved as a superseded copy, never erased.
    assert resumed["receipt"]["superseded_prior_receipt_path"]
    assert Path(resumed["receipt"]["superseded_prior_receipt_path"]).is_file()


def test_admitted_sources_contract_failure_blocks_and_keeps_the_prior_receipt(ctx: ProductionRunnerContext) -> None:
    seed_program_ledger(ctx, {"n-001": PAPER_TEXT})
    assert run_admitted(ctx)["status"] == "succeeded"
    canonical = ctx.receipt_path("corpus-inventory-index")
    prior = canonical.read_bytes()

    outcome = run_admitted(ctx, contract_status="blocked")

    assert outcome["status"] == "blocked"
    assert [blocker["code"] for blocker in outcome["receipt"]["blockers"]] == ["final_stage_contract_failed"]
    assert canonical.read_bytes() == prior
    assert re.search(r"\.attempt-", outcome["receipt_path"])


def test_admitted_sources_record_order_and_declared_builders(ctx: ProductionRunnerContext) -> None:
    seed_program_ledger(ctx, {"n-001": PAPER_TEXT})
    seen: list[str] = []

    def admit(assembly: corpus_index_stage.IndexAssembly) -> None:
        seen.append("admit")
        assembly.binding["extra_source"] = "bound"

    def build(assembly: corpus_index_stage.IndexAssembly) -> None:
        seen.append("build")
        assembly.group("extras").append(
            {"artifact_id": "extra:1", "source_family_id": "speech_transcript", "title": "A talk",
             "indexable_text": False, "source_path": "x", "storage_lane": "x", "retention_class": "x"}
        )

    stage6 = corpus_index_stage
    builders = (
        stage6.download_ledger_admission(),
        stage6.corpus_freeze_admission(CORPUS_FREEZE),
        stage6.transcription_ocr_admission(),
        stage6.RecordBuilder("extras", admit=admit, build=build),
        stage6.source_discovery_admission(SOURCE_DISCOVERY),
        stage6.main_conference_paper_records(
            build_records=lambda **kwargs: stage6.build_main_conference_paper_records(
                **kwargs, text_dir=Path("storage/corpus/paper-text"), text_lane="main-paper"
            )
        ),
    )
    outcome = corpus_index_stage.run_admitted_sources_stage(
        ledger_conventions(ctx),
        workspace=ctx.workspace,
        run_id=ctx.run_id,
        resume=True,
        full_rebuild=False,
        verify_chunk_cache=True,
        settings=corpus_index_stage.AdmittedSourcesSettings(
            inventory_receipt_type="conference_pipeline.synthetic_corpus_inventory",
            owner_function="run_corpus_inventory_index_stage",
            corpus_freeze_path=CORPUS_FREEZE,
            source_discovery_path=SOURCE_DISCOVERY,
            paper_curation_min_text_bytes=0,
            record_order=("main_papers", "extras"),
        ),
        hooks=ConferenceHooks(record_builders=builders),
    )

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    assert seen == ["admit", "build"]
    assert outcome["receipt"]["checks"]["extra_source"] == "bound"
    assert [record["artifact_id"] for record in outcome["corpus_inventory"]["artifact_records"]] == ["paper:n-001", "extra:1"]


def test_validate_admitted_binary_artifact_digest_before_magic(ctx: ProductionRunnerContext) -> None:
    archive = ctx.workspace / "storage" / "x.pdf"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"PK\x03\x04zip")

    _resolved, mismatch = corpus_index_stage.validate_admitted_binary_artifact(
        workspace=ctx.workspace, path_value=archive, expected_sha256="0" * 64, require_magic=b"%PDF-"
    )
    _resolved, impostor = corpus_index_stage.validate_admitted_binary_artifact(
        workspace=ctx.workspace, path_value=archive, expected_sha256=sha256_file(archive), require_magic=b"%PDF-"
    )
    _resolved, outside = corpus_index_stage.validate_admitted_binary_artifact(
        workspace=ctx.workspace, path_value="/etc/hosts", expected_sha256="x"
    )

    assert [item["code"] for item in mismatch] == ["artifact_provenance_mismatch"]
    assert corpus_index_stage.is_soft_impostor_rejection(impostor)
    assert outside[0]["code"] in {"out_of_workspace_path", "missing_local_path"}
    assert all(item["affected_stage"] == "corpus-inventory-index" for item in mismatch + impostor + outside)


# --------------------------------------------------------------------------
# Hook conformance and portability
# --------------------------------------------------------------------------


def test_record_builders_are_validated() -> None:
    step = corpus_index_stage.RecordBuilder("same", build=lambda _assembly: None)
    hooks = ConferenceHooks(record_builders=(step, step))

    assert any("duplicate record builder ids" in item for item in validate_hooks(hooks, ["corpus-inventory-index"]))
    assert any("no corpus-inventory-index stage" in item for item in validate_hooks(ConferenceHooks(record_builders=(step,)), ["analysis"]))
    assert corpus_index_stage.validate_record_builders([corpus_index_stage.RecordBuilder("empty")]) == [
        "record builder 'empty' implements no phase"
    ]
    with pytest.raises(ValueError, match="duplicate record builder ids"):
        corpus_index_stage._record_builders(hooks, tuple)


def test_shared_module_carries_no_conference_literal() -> None:
    text = MODULE_PATH.read_text(encoding="utf-8").lower()
    for token in ("icml", "cvpr", "iclr", "neurips", "slideslive.com", "2026"):
        assert token not in text, token


def test_both_runners_declare_their_stage6_quirks_as_hooks() -> None:
    from conference_pipeline.production.hooks import declared_hooks
    from tests.orchestrator_characterization import load_runner

    icml = declared_hooks(load_runner("icml2026"))
    cvpr = declared_hooks(load_runner("cvpr2026"))
    assert {
        "corpus-inventory-index.index_token_count",
        "corpus-inventory-index.record_builders.event_talk_maps",
        "corpus-inventory-index.record_builders.workshop_talk_annotation_bindings",
        "corpus-inventory-index.record_builders.record_titles",
        "corpus-inventory-index.record_builders.transcript_exclusions",
        "corpus-inventory-index.record_builders.workshop_talk_media",
    } <= set(icml)
    assert {
        "corpus-inventory-index.record_builders.event_media_captions",
        "corpus-inventory-index.record_builders.workshop_papers",
        "corpus-inventory-index.record_builders.main_conference_papers",
    } <= set(cvpr)
