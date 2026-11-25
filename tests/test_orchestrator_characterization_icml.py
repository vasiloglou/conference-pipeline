"""W1 characterization: pin the current ICML 2026 runner Stages 1-9.

Each test drives one stage entry point of ``scripts/run_icml2026_production.py``
in-process against a temp copy of the synthetic ``icml_shaped`` fixture (placed
at ``tmp_path/conferences/icml2026`` so the ICML runtime contracts bind), with
every network/provider/LLM/embedding/subprocess seam faked, and compares a
stable snapshot of the receipt and outputs with
``tests/golden/orchestrators/icml2026/<name>.json``.

See ``tests/orchestrator_characterization.py`` for the snapshot rules and
``ORCHESTRATOR_GOLDEN_UPDATE=1`` to re-record.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from conference_pipeline.analysis.narrative import (
    deterministic_narrative_topic_report,
    source_pack_content_sha256,
)
from conference_pipeline.indexing.embeddings import (
    build_embedding_client as _real_build_embedding_client,
)
from conference_pipeline.openreview_auth import OpenReviewAuth
from conference_pipeline.production_preflight import PLANNED_PROVIDER_PROBES
from tests.lineage_fixtures import stamp_upstream_stage_receipts
from tests.orchestrator_characterization import (
    FROZEN_NOW,
    FakeWeb,
    assert_golden,
    load_runner,
    materialize_conference_workspace,
    offline_guard,
    read_json,
    stable_snapshot,
    write_json,
)

CONFERENCE = "icml2026"
FIXTURE = "icml_shaped"
RUN_ID = "w1-icml-fixture-run"
FAKE_OPENREVIEW_TOKEN = "w1-fake-openreview-bearer-token"
FAKE_KEY = "w1-fake-provider-key-never-recorded"


# --------------------------------------------------------------------------
# Synthetic ICML-shaped web (served by FakeWeb, never a socket)
# --------------------------------------------------------------------------

ICML_SITE = "https://icml.cc"
CALENDAR_URL = f"{ICML_SITE}/virtual/2026/calendar"
OPENREVIEW_GROUP_URL = "https://openreview.net/group?id=ICML.cc/2026/Conference"
# One typed item page per declared media-bearing family plus a poster page.
ITEM_PAGES = {
    "/virtual/2026/poster/1001": ("Poster: Neural PDE Solvers for Fluid Flow", "/media/talks/poster-1001.mp4"),
    "/virtual/2026/oral/2001": ("Oral: Tool-Using Agents", "/media/talks/oral-2001.mp4"),
    "/virtual/2026/tutorial/3001": ("Tutorial: Physics-Informed Learning", "/media/talks/tutorial-3001.mp4"),
    "/virtual/2026/workshop/4001": ("Workshop: Agents", "/media/talks/workshop-4001.mp4"),
    "/virtual/2026/expo-talk/5001": ("Expo Talk: Systems", "/media/talks/expo-5001.mp4"),
    "/virtual/2026/invited-talk/6001": ("Invited Talk", "/media/talks/invited-6001.mp4"),
}


# The shared source-pack selector keeps one chunk per source artifact and the
# ICML analysis stage needs >= 3 chunks per topic, so each fixture topic gets
# two extra synthetic accepted papers on the OpenReview lane.
EXTRA_OPENREVIEW_PAPERS = (
    {"id": "q-003", "title": "Fluid Flow Surrogates with Neural PDE Solvers", "authors": ["C. Author"]},
    {"id": "q-004", "title": "Physics-Informed Fluid Flow Models", "authors": ["D. Author"]},
    {"id": "q-005", "title": "Planning with Tool-Using Agents", "authors": ["E. Author"]},
    {"id": "q-006", "title": "Evaluating Tool-Using Agents", "authors": ["F. Author"]},
)


def _fixture_papers(workspace: Path) -> list[dict[str, Any]]:
    return list(read_json(workspace / "source_inventory.json")["papers"])


def _openreview_papers(workspace: Path) -> list[dict[str, Any]]:
    return _fixture_papers(workspace) + [dict(item) for item in EXTRA_OPENREVIEW_PAPERS]


def _pdf_bytes(title: str) -> bytes:
    """A small real PDF whose text layer clears the 5,000-byte curation floor."""

    import fitz  # PyMuPDF, already a runner dependency

    doc = fitz.open()
    subject = title.lower()
    for section in range(1, 9):
        page = doc.new_page()
        body = f"{title}\nSection {section}\n" + " ".join(
            f"Paragraph {section}.{index} studies {subject} with controlled evidence, "
            f"ablations over setting {index}, and a reproducible evaluation protocol."
            for index in range(1, 13)
        )
        page.insert_textbox(fitz.Rect(50, 50, 560, 800), body, fontsize=9)
    data = doc.tobytes(garbage=4, deflate=True, no_new_id=True)
    doc.close()
    return data


def safe_venue_dir(venue: str) -> str:
    return venue.replace("/", "_")


def build_icml_web(workspace: Path, runner: ModuleType) -> FakeWeb:
    web = FakeWeb()
    calendar_links = "".join(f"<a href='{path}'>{title}</a>" for path, (title, _media) in ITEM_PAGES.items())
    web.add(CALENDAR_URL, f"<html><head><title>ICML 2026 Calendar</title></head><body>{calendar_links}</body></html>")
    web.add(
        OPENREVIEW_GROUP_URL,
        "<html><head><title>ICML 2026 Conference | OpenReview</title></head><body>venue</body></html>",
    )
    for path, (title, media) in ITEM_PAGES.items():
        web.add(
            f"{ICML_SITE}{path}",
            f"<html><head><title>{title}</title></head><body><a href='{media}'>recording</a></body></html>",
        )
        web.add(f"{ICML_SITE}{media}", f"fake-media-bytes:{media}".encode(), content_type="video/mp4")
    notes = []
    for number, paper in enumerate(_openreview_papers(workspace), start=1):
        note_id = paper["id"].replace("-", "")
        notes.append(
            {
                "id": note_id,
                "forum": note_id,
                "number": number,
                "content": {
                    "title": {"value": paper["title"]},
                    "authors": {"value": paper["authors"]},
                    "keywords": {"value": ["fixture"]},
                    "abstract": {"value": f"Fixture abstract for {paper['title']}."},
                    "venue": {"value": "ICML 2026 Poster"},
                    "venueid": {"value": runner.OPENREVIEW_ACCEPTED_VENUE_ID},
                    "pdf": {"value": f"/pdf/{note_id}.pdf"},
                },
            }
        )
        pdf = _pdf_bytes(paper["title"])
        web.add(f"https://openreview.net/pdf/{note_id}.pdf", pdf, content_type="application/pdf")
        web.add(
            f"https://api2.openreview.net/attachment?id={note_id}&name=pdf",
            pdf,
            content_type="application/pdf",
        )
    from urllib.parse import urlencode

    query = urlencode({"content.venueid": runner.OPENREVIEW_ACCEPTED_VENUE_ID, "limit": 1000, "offset": 0})
    web.add(f"{runner.OPENREVIEW_API_NOTES}?{query}", json.dumps({"notes": notes}), content_type="application/json")
    return web


def write_talk_media_scope_waiver(workspace: Path) -> Path:
    return write_json(
        workspace / "storage" / "workshops" / "workshop_talk_media_scope_waiver.json",
        {
            "schema_version": 1,
            "status": "deferred_with_explicit_scope",
            "decision": "deferred_with_explicit_scope",
            "rationale": "W1 synthetic fixture: workshop talk media acquisition is out of scope",
        },
    )


TOPIC_TITLES = {
    "01_synthetic_topic_one": ("Physics-informed learning for fluid flow", ["neural pde solvers", "fluid flow"]),
    "02_synthetic_topic_two": ("Tool-using agents", ["tool-using agents", "agents"]),
}


# The analysis stage contract (stage_owner_registry) still expects 12 topics,
# so the fixture's two topics are padded with filler topics whose keywords
# match no fixture text. The runner itself takes the topic count from the
# strategy's final_topics (W5 Phase B).
FILLER_TOPICS = (
    "kappa lattice",
    "lambda quarry",
    "mu orchard",
    "nu harbor",
    "xi meadow",
    "omicron canyon",
    "pi glacier",
    "rho tundra",
    "sigma lagoon",
    "tau prairie",
)


def write_topic_strategy(
    workspace: Path,
    *,
    topic_count: int = 12,
    title_overrides: dict[str, tuple[str, list[str]]] | None = None,
    declared_topic_count: int | None = None,
) -> Path:
    """Runner-format topic strategy built from the fixture's topic ids."""

    topics = list(read_json(workspace / "analysis" / "topics.json"))
    for index, filler in enumerate(FILLER_TOPICS, start=len(topics) + 1):
        topics.append({"id": f"{index:02d}_{filler.replace(' ', '_')}"})
        TOPIC_TITLES.setdefault(topics[-1]["id"], (filler.title(), [filler]))
    topics = topics[:topic_count]
    final_topics = []
    for rank, topic in enumerate(topics, start=1):
        title, patterns = (title_overrides or {}).get(topic["id"]) or TOPIC_TITLES[topic["id"]]
        final_topics.append(
            {
                "rank": rank,
                "topic_id": topic["id"],
                "title": title,
                "starter_queries": [title],
                "evidence_title_patterns": patterns,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "succeeded",
        "mode": "live",
        "final_topics": final_topics,
        "provider_receipt_paths": [f"storage/workflow-runs/{RUN_ID}/provider-calls/topic-strategy.json"],
    }
    if declared_topic_count is not None:
        payload["topic_count"] = declared_topic_count
    write_json(workspace / "storage" / "analysis" / "topic-strategy" / RUN_ID / "topic_strategy_receipt.json", payload)
    return write_json(workspace / "analysis" / "topic_strategy.json", payload)


def write_source_intake(workspace: Path) -> Path:
    """Runner-format source-intake inventory derived from the fixture."""

    families = [
        ("icml-calendar", "ICML Calendar", CALENDAR_URL),
        ("openreview-venue", "OpenReview Venue", OPENREVIEW_GROUP_URL),
    ]
    payload = {
        "schema_version": 1,
        "conference": {"slug": CONFERENCE, "name": "Synthetic ICML-shaped fixture"},
        "source_urls": {"calendar": CALENDAR_URL, "openreview_venue": OPENREVIEW_GROUP_URL},
        "source_families": [
            {
                "id": family_id,
                "label": label,
                "source_url": url,
                "source_kind": "html",
                "expected_coverage": "fixture coverage",
                "status": "present",
                "support_label": "fixture",
                "live_proof_status": "fixture",
                "auth_assumption": "public",
                "notes": "synthetic W1 fixture",
            }
            for family_id, label, url in families
        ],
    }
    terms = read_json(workspace / "analysis" / "physics_priority_terms.json")["terms"]
    write_json(workspace / "storage" / "source-intake" / "physics_priority.json", {"keyword_terms": terms})
    return write_json(workspace / "storage" / "source-intake" / "source_inventory.json", payload)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class IcmlHarness:
    def __init__(self, runner: ModuleType, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.runner = runner
        self.workspace = workspace
        self.monkeypatch = monkeypatch
        self.web = build_icml_web(workspace, runner)
        monkeypatch.setattr(runner, "urlopen", self.web)
        monkeypatch.setattr(runner, "utc_now", lambda: FROZEN_NOW)
        monkeypatch.setattr(
            runner,
            "load_openreview_auth",
            lambda *_a, **_k: OpenReviewAuth(
                FAKE_OPENREVIEW_TOKEN, "authenticated", ["OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD"]
            ),
        )
        monkeypatch.setattr(runner, "OPENREVIEW_AUTH", OpenReviewAuth(None, "unauthenticated", []))
        monkeypatch.setattr(runner, "run_workshop_discovery_and_download", self.fake_workshop_acquisition)
        monkeypatch.setattr(runner.time, "sleep", lambda *_a, **_k: None)
        self.workshop_calls: list[dict[str, Any]] = []

    def fake_workshop_acquisition(self, workspace: Path, **kwargs: Any) -> dict[str, Any]:
        self.workshop_calls.append({"workspace": str(workspace), **kwargs})
        for paper in _fixture_papers(self.workspace):
            if paper.get("venue"):
                venue_dir = safe_venue_dir(paper["venue"])
                target = self.workspace / "storage" / "workshops" / "paper-downloads" / venue_dir / f"{paper['id']}.pdf"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_pdf_bytes(paper["title"]))
        return {
            "parent_venue": kwargs.get("parent_venue"),
            "rate_limit_rps": kwargs.get("rate_limit_rps"),
            "status": "succeeded",
            "discovery": {"workshop_count": 1},
            "download": {
                "output_dir": "storage/workshops/paper-downloads",
                "totals": {"available": 1, "succeeded": 1, "failed": 0, "resume_skipped": 0},
            },
        }

    def args(self, stage: str, *extra: str) -> Any:
        return self.runner.parse_args(
            [
                "--workspace",
                str(self.workspace),
                "--run-id",
                RUN_ID,
                "--stage",
                stage,
                "--no-env-file",
                "--min-free-gb",
                "0",
                "--source-timeout-seconds",
                "1",
                "--download-timeout-seconds",
                "1",
                "--download-retry-count",
                "1",
                "--item-page-worker-count",
                "1",
                "--paper-pdf-worker-count",
                "1",
                "--item-page-auth-retry-passes",
                "0",
                *extra,
            ]
        )

    def run(self, stage: str, *extra: str) -> tuple[int, dict[str, Any]]:
        stage_obj = self.runner.STAGES[stage]
        handler = {
            "runner-preflight": self.runner.run_runner_preflight,
            "source-discovery": self.runner.run_source_discovery,
            "corpus-freeze": self.runner.run_corpus_freeze,
            "download-corpus": self.runner.run_download_corpus,
            "transcription-ocr": self.runner.run_transcription_ocr,
            "corpus-inventory-index": self.runner.run_corpus_inventory_index,
            "exhaustive-topic-extraction": self.runner.run_exhaustive_topic_extraction_stage,
            "analysis": self.runner.run_analysis,
            "qa-gap-fill-synthesis": self.runner.run_qa_gap_fill_synthesis,
            "podcast": self.runner.run_podcast,
        }[stage]
        return handler(self.args(stage, *extra), stage_obj)

    def run_stage_run(self, stage: str, *extra: str) -> tuple[int, dict[str, Any]]:
        """Run a stage entry and normalize its result to ``(exit_code, summary)``."""

        from conference_pipeline.production import orchestrator as production_orchestrator

        run = production_orchestrator.normalize_stage_result(self.run(stage, *extra))
        exit_code = run.exit_code if run.exit_code is not None else (0 if run.status == "succeeded" else 1)
        return exit_code, run.summary_payload()

    def receipt(self, stage: str) -> dict[str, Any]:
        stage_obj = self.runner.STAGES[stage]
        return read_json(self.runner.receipt_path(self.workspace, RUN_ID, stage_obj))

    def receipt_file(self, stage: str) -> Path:
        return self.runner.receipt_path(self.workspace, RUN_ID, self.runner.STAGES[stage])

    def use_fake_embeddings(self) -> None:
        """Local deterministic ``fake`` embedding provider; anything else fails."""

        self.monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
        self.monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")

        def only_fake(config: dict[str, Any]) -> Any:
            assert config.get("provider") == "fake", config
            return _real_build_embedding_client(config)

        self.monkeypatch.setattr(self.runner, "build_embedding_client", only_fake)

    def set_provider_env(self) -> None:
        for name in self.runner.REQUIRED_PROVIDER_ENV:
            self.monkeypatch.setenv(name, FAKE_KEY)

    def assert_secret_free(self) -> None:
        for path in self.workspace.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".md", ".jsonl", ".txt"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                assert FAKE_KEY not in text, path
                assert FAKE_OPENREVIEW_TOKEN not in text, path


@pytest.fixture
def icml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = load_runner(CONFERENCE)
    workspace = materialize_conference_workspace(FIXTURE, tmp_path, CONFERENCE)
    with offline_guard(monkeypatch, runner):
        harness = IcmlHarness(runner, workspace, monkeypatch)
        yield harness


def governed_model_literals(runner: ModuleType) -> dict[str, str]:
    """Registry-governed model ids -> placeholders (a catalog upgrade is not drift)."""

    import sys

    from conference_pipeline.model_governance import default_model

    literals = {
        sys.executable: "<PYTHON>",
        runner.DEFAULT_DEEP_NARRATIVE_MODEL: "<MODEL:analysis.deep_narrative_synthesis>",
    }
    for task_id in (
        runner.SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID,
        runner.PODCAST_SCRIPT_MODEL_TASK_ID,
        runner.PODCAST_TTS_TASK_ID,
    ):
        try:
            literals.setdefault(default_model(task_id), f"<MODEL:{task_id}>")
        except Exception:  # noqa: BLE001 - a task without a default model has no literal
            continue
    return literals


def _snap(harness: IcmlHarness, stage: str, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("literals", governed_model_literals(harness.runner))
    kwargs.setdefault("json_digest_fns", {"source_pack_content": source_pack_content_sha256})
    # Provenance records the digest of the in-memory pack, which carries the
    # (path-bearing) corpus inventory and index digest shas.
    kwargs.setdefault("derived_digest_keys", ("source_pack_sha256",))
    return stable_snapshot(harness.receipt(stage), workspace=harness.workspace, **kwargs)


# Decision D1 (2026-09-30): ICML runs the shared approval / estimate / capacity
# preflight. The prepared estimate digests the absolute workspace path, so every
# digest derived from it differs per temp dir.
PREFLIGHT_DERIVED_DIGESTS = ("estimate_digest", "source_pack_sha256")
PREFLIGHT_FREE_BYTES = 10**12


def _preflight_snap(harness: IcmlHarness) -> dict[str, Any]:
    return _snap(harness, "runner-preflight", derived_digest_keys=PREFLIGHT_DERIVED_DIGESTS)


def _prepare_preflight_authority(harness: IcmlHarness) -> dict[str, Any]:
    """The no-provider preparation an operator runs before live preflight."""

    write_json(
        harness.workspace / "storage" / "preflight-storage-policy.json",
        {
            "corpus_scope": harness.runner.PREFLIGHT_SETTINGS.corpus_scope,
            "estimator_version": "fixture-v1",
            "remaining_families": {"paper_pdf": 700, "media": 300},
            "safety_margin_bytes": 100,
        },
    )
    prepared = harness.runner.prepare_preflight_estimate(harness.workspace, RUN_ID)
    harness.runner.prepare_preflight_storage_estimate(harness.workspace, RUN_ID)
    return prepared


def _fake_preflight_environment(harness: IcmlHarness, probed: list[str] | None = None) -> None:
    import shutil

    import conference_pipeline.costs.estimator as estimator

    # Page counts come from pypdf, never a platform metadata subprocess.
    harness.monkeypatch.setattr(estimator, "_run_text_command", lambda *_a, **_k: None)
    harness.monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: type("DiskUsage", (), {"free": PREFLIGHT_FREE_BYTES, "total": PREFLIGHT_FREE_BYTES})()
    )

    def probe(spec: Any) -> Any:
        from conference_pipeline.production_preflight import ProviderProbeResult

        if probed is not None:
            probed.append(spec.provider)
        return ProviderProbeResult.succeeded(spec)

    harness.monkeypatch.setattr(harness.runner, "preflight_probe", probe)


def _set_probe_credentials(harness: IcmlHarness) -> None:
    harness.set_provider_env()
    for name in harness.runner.PREFLIGHT_CREDENTIAL_ENV:
        harness.monkeypatch.setenv(name, FAKE_KEY)


# --------------------------------------------------------------------------
# Stage 1: runner-preflight
# --------------------------------------------------------------------------


def test_icml_runner_preflight_success_receipt(icml: IcmlHarness) -> None:
    probed: list[str] = []
    _fake_preflight_environment(icml, probed)
    prepared = _prepare_preflight_authority(icml)
    _set_probe_credentials(icml)

    exit_code, summary = icml.run_stage_run("runner-preflight")

    assert exit_code == 0, summary
    receipt = icml.receipt("runner-preflight")
    assert receipt["checks"]["spend_authority"]["mode"] == "automatic_under_usd_800"
    assert receipt["checks"]["prepared_estimate"]["provider_stage_projection"] == prepared["provider_stage_projection"]
    assert all(cost > 0 or key.startswith("google_notebooklm:") for key, cost in prepared["provider_stage_projection"].items())
    assert sorted(probed) == sorted(spec.provider for spec in PLANNED_PROVIDER_PROBES)
    assert_golden(CONFERENCE, "stage01_runner_preflight__succeeded", _preflight_snap(icml))
    icml.assert_secret_free()


def test_icml_runner_preflight_missing_authority_blocks_before_any_probe(icml: IcmlHarness) -> None:
    _fake_preflight_environment(icml)
    icml.monkeypatch.setattr(icml.runner, "preflight_probe", _must_not_rebuild)

    exit_code, summary = icml.run_stage_run("runner-preflight")

    assert exit_code == 1
    assert summary["status"] == "blocked"
    assert_golden(CONFERENCE, "stage01_runner_preflight__blocked_missing_authority", _preflight_snap(icml))


def test_icml_runner_preflight_missing_credentials_blocks(icml: IcmlHarness) -> None:
    _fake_preflight_environment(icml)
    _prepare_preflight_authority(icml)
    for name in (*icml.runner.REQUIRED_PROVIDER_ENV, *icml.runner.PREFLIGHT_CREDENTIAL_ENV):
        icml.monkeypatch.delenv(name, raising=False)

    exit_code, summary = icml.run_stage_run("runner-preflight")

    assert exit_code == 1
    assert summary["status"] == "blocked"
    assert_golden(CONFERENCE, "stage01_runner_preflight__blocked_missing_credentials", _preflight_snap(icml))


def test_icml_runner_preflight_resume_reruns_the_preflight(icml: IcmlHarness) -> None:
    """D1: a preflight never resumes; --resume re-checks (preflight_resume_policy rerun_required)."""

    probed: list[str] = []
    _fake_preflight_environment(icml, probed)
    _prepare_preflight_authority(icml)
    _set_probe_credentials(icml)
    assert icml.run_stage_run("runner-preflight")[0] == 0
    first_probe_count = len(probed)

    exit_code, summary = icml.run_stage_run("runner-preflight", "--resume")

    assert exit_code == 0
    assert summary["status"] == "succeeded"
    assert len(probed) == 2 * first_probe_count
    assert icml.receipt("runner-preflight")["preflight_resume_policy"] == "rerun_required"


def test_icml_prepare_commands_run_through_main(icml: IcmlHarness, capsys: pytest.CaptureFixture[str]) -> None:
    _fake_preflight_environment(icml)
    base = ["--workspace", str(icml.workspace), "--run-id", RUN_ID]
    assert icml.runner.main([*base, "--prepare-preflight-estimate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["corpus_scope"] == "all-discovered-icml2026"
    assert (icml.runner.run_dir(icml.workspace.resolve(), RUN_ID) / "preflight-full-run-estimate.json").is_file()
    # No storage policy yet: the capacity input cannot be bound.
    assert icml.runner.main([*base, "--prepare-storage-estimate"]) == 1
    assert "cannot prepare ICML 2026 storage estimate" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        icml.runner.parse_args([*base, "--prepare-preflight-estimate", "--stage", "runner-preflight"])


def _must_not_rebuild(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("resume re-ran the stage instead of accepting the persisted receipt")


# --------------------------------------------------------------------------
# Stage 2: source-discovery
# --------------------------------------------------------------------------


def run_through(icml: IcmlHarness, last_stage: str) -> None:
    order = [
        "source-discovery",
        "corpus-freeze",
        "download-corpus",
        "transcription-ocr",
        "corpus-inventory-index",
        "exhaustive-topic-extraction",
        "analysis",
        "qa-gap-fill-synthesis",
        "podcast",
    ]
    write_source_intake(icml.workspace)
    for stage in order:
        if stage == "transcription-ocr":
            prepare_transcription_inputs(icml)
        if stage == "corpus-inventory-index":
            icml.use_fake_embeddings()
        if stage == "exhaustive-topic-extraction":
            write_topic_strategy(icml.workspace)
        if stage == "analysis":
            write_analysis_substep_receipts(icml)
            run_analysis_success(icml)
            if stage == last_stage:
                return
            continue
        if stage == "qa-gap-fill-synthesis":
            run_stage8_success(icml)
            if stage == last_stage:
                return
            continue
        if stage == "podcast":
            run_podcast_success(icml)
            return
        exit_code, summary = icml.run(stage)
        assert exit_code == 0, (stage, summary)
        if stage == last_stage:
            return


def test_icml_source_discovery_success_receipt(icml: IcmlHarness) -> None:
    write_source_intake(icml.workspace)
    exit_code, summary = icml.run("source-discovery")

    assert exit_code == 0, summary
    assert icml.web.calls == [CALENDAR_URL, OPENREVIEW_GROUP_URL]
    assert_golden(CONFERENCE, "stage02_source_discovery__succeeded", _snap(icml, "source-discovery"))


def test_icml_source_discovery_missing_intake_blocks(icml: IcmlHarness) -> None:
    exit_code, _summary = icml.run("source-discovery")

    assert exit_code == 1
    assert icml.web.calls == []
    assert_golden(CONFERENCE, "stage02_source_discovery__blocked_missing_intake", _snap(icml, "source-discovery"))


# --------------------------------------------------------------------------
# Stage 3: corpus-freeze
# --------------------------------------------------------------------------


def test_icml_corpus_freeze_success_receipt(icml: IcmlHarness) -> None:
    run_through(icml, "source-discovery")
    calls_before = list(icml.web.calls)
    exit_code, summary = icml.run("corpus-freeze")

    assert exit_code == 0, summary
    assert icml.web.calls == calls_before, "corpus-freeze must not touch the network"
    assert_golden(CONFERENCE, "stage03_corpus_freeze__succeeded", _snap(icml, "corpus-freeze"))


# --------------------------------------------------------------------------
# Stage 4: download-corpus
# --------------------------------------------------------------------------


def test_icml_download_corpus_receipt(icml: IcmlHarness) -> None:
    run_through(icml, "corpus-freeze")
    exit_code, summary = icml.run("download-corpus")

    download = read_json(icml.workspace / icml.runner.DOWNLOAD_CORPUS_INVENTORY)
    assert icml.workshop_calls and icml.workshop_calls[0]["parent_venue"] == icml.runner.OPENREVIEW_WORKSHOP_PARENT_VENUE
    # Bearer auth: PDFs go through the authenticated attachment endpoint.
    assert any("api2.openreview.net/attachment" in url for url in icml.web.calls)
    assert_golden(
        CONFERENCE,
        "stage04_download_corpus__authenticated",
        _snap(
            icml,
            "download-corpus",
            extra_output_paths=[icml.runner.DOWNLOAD_CORPUS_OPENREVIEW, "cache/acquisition-downloads"],
        ),
    )
    icml.assert_secret_free()
    assert (exit_code, download["status"]) in {(0, "succeeded"), (1, "blocked")}


# --------------------------------------------------------------------------
# Stage 5: transcription-ocr
# --------------------------------------------------------------------------


def test_icml_transcription_ocr_blocks_without_workshop_talk_media(icml: IcmlHarness) -> None:
    run_through(icml, "download-corpus")
    exit_code, summary = icml.run("transcription-ocr")

    assert exit_code == 1
    assert summary["status"] == "blocked"
    assert_golden(
        CONFERENCE,
        "stage05_transcription_ocr__blocked_workshop_talk_media_missing",
        _snap(icml, "transcription-ocr"),
    )


def test_icml_transcription_ocr_blocks_on_unwaived_speech_inputs(icml: IcmlHarness) -> None:
    run_through(icml, "download-corpus")
    write_talk_media_scope_waiver(icml.workspace)
    exit_code, summary = icml.run("transcription-ocr")

    assert exit_code == 1
    assert summary["stages"][0]["checks"]["blockers"][0]["code"] == "speech_inputs_present_provider_path_required"
    assert_golden(
        CONFERENCE,
        "stage05_transcription_ocr__blocked_speech_provider_required",
        _snap(icml, "transcription-ocr", extra_output_paths=["storage/transcription-ocr"]),
    )


def prepare_transcription_inputs(icml: IcmlHarness) -> None:
    write_talk_media_scope_waiver(icml.workspace)
    write_json(
        icml.workspace / icml.runner.TRANSCRIPTION_OCR_OPERATOR_WAIVERS,
        {
            "schema_version": 1,
            "waivers": [
                {
                    "code": "speech_inputs_present_provider_path_required",
                    "decision": "waived",
                    "reason": "W1 synthetic fixture: no ASR provider in characterization tests",
                    "operator": "w1-fixture",
                }
            ],
        },
    )


def test_icml_transcription_ocr_local_lanes_receipt(icml: IcmlHarness) -> None:
    run_through(icml, "download-corpus")
    prepare_transcription_inputs(icml)
    exit_code, summary = icml.run("transcription-ocr")

    assert exit_code == 0, summary
    assert_golden(
        CONFERENCE,
        "stage05_transcription_ocr__local_lanes_speech_waived",
        _snap(icml, "transcription-ocr", extra_output_paths=["storage/transcription-ocr"]),
    )


# --------------------------------------------------------------------------
# Stage 6: corpus-inventory-index
# --------------------------------------------------------------------------

# Chroma's persisted SQLite/HNSW files carry random ids; the stores are pinned
# through their chunk sidecars, manifests and index digest instead.
INDEX_SKIP_DIRS = ("chroma",)


def test_icml_corpus_inventory_index_fake_embedding_receipt(icml: IcmlHarness) -> None:
    run_through(icml, "transcription-ocr")
    icml.use_fake_embeddings()
    exit_code, summary = icml.run("corpus-inventory-index")
    print(json.dumps(summary["stages"][0].get("error"), default=str))

    assert_golden(
        CONFERENCE,
        "stage06_corpus_inventory_index__fake_embeddings",
        _snap(
            icml,
            "corpus-inventory-index",
            extra_output_paths=[f"storage/vector-store/{RUN_ID}"],
            skip_dirs=INDEX_SKIP_DIRS,
        ),
    )
    assert exit_code == 0, summary


# --------------------------------------------------------------------------
# Stage 7A: exhaustive-topic-extraction
# --------------------------------------------------------------------------


def test_icml_exhaustive_topic_extraction_receipt(icml: IcmlHarness) -> None:
    run_through(icml, "corpus-inventory-index")
    write_topic_strategy(icml.workspace)
    exit_code, summary = icml.run("exhaustive-topic-extraction")
    print(json.dumps(summary["stages"][0].get("error"), default=str), summary["status"])

    assert_golden(
        CONFERENCE,
        "stage07a_exhaustive_topic_extraction__succeeded",
        _snap(icml, "exhaustive-topic-extraction", extra_output_paths=["storage/extractions"]),
    )
    assert exit_code == 0, summary


# --------------------------------------------------------------------------
# Stage 7: analysis (gate chain + deterministic source packs)
# --------------------------------------------------------------------------


def test_icml_analysis_blocks_on_a_topic_strategy_count_mismatch(icml: IcmlHarness) -> None:
    """W5 Phase B: the expected topic count comes from the strategy; a strategy
    whose declared topic_count disagrees with its final_topics blocks."""

    run_through(icml, "exhaustive-topic-extraction")
    write_topic_strategy(icml.workspace, topic_count=2, declared_topic_count=12)
    exit_code, summary = icml.run("analysis")

    assert exit_code == 1
    assert summary["stages"][0]["checks"]["blocker_code"] == "analysis_topic_strategy_topic_count_mismatch"
    assert_golden(CONFERENCE, "stage07_analysis__blocked_topic_strategy_count", _snap(icml, "analysis"))


def test_icml_analysis_takes_the_topic_count_from_the_strategy(icml: IcmlHarness) -> None:
    """A two-topic strategy passes the topic gate (it used to require exactly 12)."""

    run_through(icml, "exhaustive-topic-extraction")
    write_topic_strategy(icml.workspace, topic_count=2, declared_topic_count=2)
    exit_code, summary = icml.run("analysis")

    checks = summary["stages"][0]["checks"]
    assert exit_code == 1
    assert checks["topic_strategy_topic_count"] == 2
    assert checks["blocker_code"] == "analysis_llm_full_read_extraction_required"


def test_icml_analysis_blocks_before_spend_without_full_read_extraction(icml: IcmlHarness) -> None:
    run_through(icml, "exhaustive-topic-extraction")
    exit_code, summary = icml.run("analysis")

    assert exit_code == 1
    assert summary["stages"][0]["checks"]["blocker_code"] == "analysis_llm_full_read_extraction_required"
    assert_golden(CONFERENCE, "stage07_analysis__blocked_llm_full_read_required", _snap(icml, "analysis"))


def write_analysis_substep_receipts(icml: IcmlHarness, *, topic_count: int = 12) -> None:
    """Synthetic 14A full-read / 07B-07H substep receipts meeting the analysis contract.

    These are the post-full-read chain scripts' receipts (already shared owners,
    not lifted in W5); stamped with fixture lineage as a real chain would.
    """

    run_path = icml.runner.run_dir(icml.workspace, RUN_ID)
    write_json(
        run_path / "14A-llm-full-read-extraction-receipt.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "scope": "all",
            "limit": None,
            "model": "w1-fixture-local-model",
            "checks": {
                "completion_state": "complete",
                "expected_unit_count": 3,
                "completed_unit_count": 3,
                "failed_count": 0,
            },
        },
    )
    write_json(
        run_path / "07D-topic-reconciliation.json",
        {"schema_version": 1, "status": "succeeded", "source_packs_enriched": True, "topic_count": topic_count},
    )
    write_json(
        run_path / "07E-topic-reference-validation.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "topic_count": topic_count,
            "checks": {
                "completed_topic_count": topic_count,
                "ledger_entry_count": 3,
                "citation_count": 3,
                "invalid_reference_count": 0,
                "unknown_reference_count": 0,
                "title_mismatch_count": 0,
                "bare_reference_warning_count": 0,
            },
        },
    )
    write_json(
        run_path / "07H-topic-reconciliation-reference-workflow.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "checks": {
                "final_validation_status": "succeeded",
                "final_topic_count": topic_count,
                "final_completed_topic_count": topic_count,
                "final_ledger_entry_count": 3,
                "final_citation_count": 3,
                "final_invalid_reference_count": 0,
                "final_unknown_reference_count": 0,
                "final_title_mismatch_count": 0,
                "final_bare_reference_warning_count": 0,
            },
        },
    )
    for name in ("07B-deep-dive-question-selection.json", "07C-dual-deep-dive-research-lanes.json"):
        write_json(run_path / name, {"schema_version": 1, "status": "succeeded", "provider_calls_made": True})
    # Chain outputs the analysis runtime contract globs for.
    topic_ids = [str(item["topic_id"]) for item in read_json(icml.workspace / "analysis" / "topic_strategy.json")["final_topics"]]
    full_read = icml.workspace / "storage" / "extractions" / RUN_ID / "llm-full-read"
    write_json(full_read / "by-unit" / "unit-0001.json", {"unit_id": "unit-0001", "input_kind": "paper_abstract"})
    for topic_id in topic_ids:
        items = [
            {
                "unit_id": f"fr-{topic_id}-{index}",
                "input_kind": "paper_abstract",
                "confidence": 0.9,
                "important": True,
                "summary": f"Fixture full-read summary {index} for {topic_id}.",
                "reason": "fixture",
                "evidence_phrases": [f"fixture phrase {index}"],
                "source_artifact_path": "storage/download-corpus/openreview_accepted_papers.json",
                "metadata": {"title": f"Fixture unit {index}", "paper_id": f"fixture-{index}"},
            }
            for index in range(1, 6)
        ]
        write_json(full_read / "by-topic" / f"{topic_id}.json", {"topic_id": topic_id, "unit_count": len(items), "items": items})
        reconciliation = icml.workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md"
        reconciliation.parent.mkdir(parents=True, exist_ok=True)
        reconciliation.write_text(f"# {topic_id}\n\nFixture reconciliation.\n", encoding="utf-8")
        write_json(icml.workspace / "analysis" / "topic_reference_ledgers" / f"{topic_id}.json", {"topic_id": topic_id, "entries": []})
        write_json(
            icml.workspace / "analysis" / "topic_reference_validation" / f"{topic_id}.json",
            {"topic_id": topic_id, "status": "succeeded"},
        )
    stamp_upstream_stage_receipts(run_path)


def test_icml_analysis_builds_source_packs_then_blocks_without_cost_approval(icml: IcmlHarness) -> None:
    run_through(icml, "exhaustive-topic-extraction")
    write_analysis_substep_receipts(icml)
    exit_code, summary = icml.run("analysis")
    checks = summary["stages"][0]["checks"]
    print(checks.get("blocker_code"), summary["stages"][0].get("error"))

    assert exit_code == 1
    # W5 Phase B: blocked before the renderer, so no placeholder reports.
    assert checks["blocker_code"] == "analysis_provider_cost_approval_required"
    assert "narrative_status" not in checks
    assert not list((icml.workspace / "analysis").glob("[0-9][0-9]_*.md"))
    assert_golden(
        CONFERENCE,
        "stage07_analysis__source_packs_then_blocked_no_cost_approval",
        _snap(icml, "analysis", extra_output_paths=["analysis/narrative_source_packs", "analysis/provenance/query_plans"]),
    )


def test_icml_analysis_identical_source_packs_write_a_blocked_receipt(icml: IcmlHarness) -> None:
    """W5 Phase B: identical topic selections block with a receipt and zero
    spend (they used to escape as an uncaught RuntimeError with no receipt)."""

    run_through(icml, "exhaustive-topic-extraction")
    same = ("Neural PDE solvers", ["neural pde solvers"])
    write_topic_strategy(
        icml.workspace, title_overrides={"01_synthetic_topic_one": same, "02_synthetic_topic_two": same}
    )
    write_analysis_substep_receipts(icml)

    exit_code, summary = icml.run("analysis", "--provider-cost-approved")

    checks = summary["stages"][0]["checks"]
    assert (exit_code, summary["status"]) == (1, "blocked")
    assert checks["blocker_code"] == "analysis_source_packs_identical"
    assert checks["identical_source_pack_groups"] == [["01_synthetic_topic_one", "02_synthetic_topic_two"]]
    assert checks["provider_calls_made"] is False
    assert summary["stages"][0]["blocker_receipt_path"]
    assert_golden(CONFERENCE, "stage07_analysis__blocked_identical_source_packs", _snap(icml, "analysis"))


class FakeNarrativeClient:
    """Stand-in for the paid deep-narrative client: returns the provider-free
    deterministic report for the source pack named in the prompt."""

    def __init__(self, workspace: Path, model: str) -> None:
        self.workspace = workspace
        self.model = model  # the governed default: Stage 8 checks it was "promoted"
        self.request_count = 0
        self.prompts: list[str] = []

    @classmethod
    def installer(cls, workspace: Path, model: str) -> type:
        client = cls(workspace, model)

        class _Factory:
            instance = client

            @classmethod
            def from_env(cls) -> FakeNarrativeClient:
                return client

        return _Factory

    def synthesize(self, *, kind: str, system_prompt: str, user_prompt: str) -> Any:
        self.request_count += 1
        self.prompts.append(kind)
        packs = sorted((self.workspace / "analysis" / "narrative_source_packs").glob("*.json"))
        for path in packs:
            pack = read_json(path)
            if str(pack.get("research_id") or "") in user_prompt:
                from types import SimpleNamespace

                return SimpleNamespace(text=deterministic_narrative_topic_report(pack))
        raise AssertionError("fake narrative client could not match the prompt to a source pack")


def enrich_source_packs_like_topic_reconciliation(icml: IcmlHarness) -> None:
    """Emulate the 07D chain: an earlier analysis pass built the packs, and
    topic reconciliation enriched them before the paid pass re-runs analysis.

    Packs come from the runner's own deterministic ``build_analysis_source_packs``.
    """

    runner = icml.runner
    workspace = icml.workspace
    stage6 = read_json(runner.receipt_path(workspace, RUN_ID, runner.STAGES["corpus-inventory-index"]))
    index_digest = read_json(workspace / runner.VECTOR_STORE_DIR / RUN_ID / "index_digest.json")
    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=RUN_ID,
        stage6_receipt=stage6,
        index_digest=index_digest,
        topic_specs=runner.load_analysis_topic_specs(workspace, run_id=RUN_ID),
    )
    for path in sorted((workspace / runner.ANALYSIS_SOURCE_PACK_DIR).glob("*.json")):
        pack = read_json(path)
        pack["topic_reconciliation"] = {
            "status": "succeeded",
            "model": "w1-fixture-reconciliation",
            "reference_validation_status": "succeeded",
            "reference_validation_counts": {
                "citation_count": 3,
                "ledger_entry_count": 3,
                "invalid_reference_count": 0,
                "unknown_reference_count": 0,
                "title_mismatch_count": 0,
                "bare_reference_warning_count": 0,
            },
        }
        write_json(path, pack)


def run_analysis_success(icml: IcmlHarness) -> FakeNarrativeClient:
    enrich_source_packs_like_topic_reconciliation(icml)
    factory = FakeNarrativeClient.installer(icml.workspace, icml.runner.DEFAULT_DEEP_NARRATIVE_MODEL)
    icml.monkeypatch.setattr(icml.runner, "OpenAIDeepNarrativeClient", factory)
    exit_code, summary = icml.run("analysis", "--provider-cost-approved")
    assert exit_code == 0, summary["stages"][0].get("error")
    assert summary["status"] == "succeeded", summary["stages"][0].get("stage_contract")
    return factory.instance


def test_icml_analysis_success_with_fake_narrative_client(icml: IcmlHarness) -> None:
    run_through(icml, "exhaustive-topic-extraction")
    write_analysis_substep_receipts(icml)
    client = run_analysis_success(icml)

    assert client.request_count == 12
    assert_golden(
        CONFERENCE,
        "stage07_analysis__succeeded_fake_narrative",
        _snap(
            icml,
            "analysis",
            extra_output_paths=[
                "analysis/narrative_source_packs",
                "analysis/provenance",
                "analysis/slide_briefs",
                "analysis/analysis_manifest.json",
            ],
        ),
    )


def test_icml_analysis_exit_code_follows_a_contract_blocked_receipt(
    icml: IcmlHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    """W5 Phase B: when the final stage contract blocks the success receipt,
    run_analysis returns 1 (it used to return 0); the CLI already exited 1
    through the reporter and its output is unchanged."""

    run_through(icml, "exhaustive-topic-extraction")
    write_analysis_substep_receipts(icml)
    enrich_source_packs_like_topic_reconciliation(icml)
    icml.monkeypatch.setattr(
        icml.runner,
        "OpenAIDeepNarrativeClient",
        FakeNarrativeClient.installer(icml.workspace, icml.runner.DEFAULT_DEEP_NARRATIVE_MODEL),
    )
    blocked_contract = SimpleNamespace(
        status="blocked",
        blockers=[{"code": "w5_fixture_contract_blocked"}],
        to_dict=lambda: {"status": "blocked", "blockers": [{"code": "w5_fixture_contract_blocked"}]},
    )
    icml.monkeypatch.setattr(icml.runner, "verify_stage_contract", lambda **_kwargs: blocked_contract)

    exit_code, summary = icml.run("analysis", "--provider-cost-approved")

    assert (exit_code, summary["status"]) == (1, "blocked")
    assert summary["stages"][0]["error"] == "runtime stage contract verification failed"

    argv = [
        "--workspace", str(icml.workspace), "--run-id", RUN_ID, "--stage", "analysis",
        "--no-env-file", "--provider-cost-approved",
    ]
    capsys.readouterr()
    assert icml.runner.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"ICML 2026 production halted at analysis; summary: {icml.runner.summary_path(icml.workspace, RUN_ID)}\n"
    )


def test_icml_analysis_resume_returns_the_verified_receipt_without_spend(icml: IcmlHarness) -> None:
    """W5 Phase B: --resume over a verifying receipt never re-enters the paid renderer."""

    run_through(icml, "exhaustive-topic-extraction")
    write_analysis_substep_receipts(icml)
    run_analysis_success(icml)
    before = icml.receipt_file("analysis").read_bytes()

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("paid analysis path re-entered on resume")

    icml.monkeypatch.setattr(icml.runner, "OpenAIDeepNarrativeClient", SimpleNamespace(from_env=_forbidden))
    icml.monkeypatch.setattr(icml.runner, "render_narrative_outputs_from_source_packs", _forbidden)

    exit_code, summary = icml.run("analysis", "--resume")

    assert (exit_code, summary["status"]) == (0, "succeeded")
    assert icml.receipt_file("analysis").read_bytes() == before


# --------------------------------------------------------------------------
# Stage 8: qa-gap-fill-synthesis
# --------------------------------------------------------------------------


class FakeStage8Client:
    """Governed-selection stand-in: registry provider/model, canned long text."""

    def __init__(self, runner: ModuleType) -> None:
        from conference_pipeline.model_governance import default_model, default_provider

        task = runner.SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID
        self.provider = default_provider(task)
        self.model = default_model(task)
        self.calls: list[str] = []

    def synthesize(self, *, kind: str, system_prompt: str, user_prompt: str) -> Any:
        from types import SimpleNamespace

        assert "Citation pool" in user_prompt
        self.calls.append(kind)
        return SimpleNamespace(
            text=(f"# {kind}\n\nValidated fixture synthesis paragraph for {kind}.\n" * 120),
            input_tokens=100,
            output_tokens=50,
        )


# The governed synthesis model/price come from the model registry; a catalog
# upgrade must not read as a stage behaviour change.
STAGE8_MASK = (
    "checks.synthesis_provider",
    "checks.estimated_cost_usd",
    "checks.synthesis_cost_usd",
    "checks.token_usage.estimated_cost_usd",
)


def run_stage8_success(icml: IcmlHarness) -> FakeStage8Client:
    client = FakeStage8Client(icml.runner)
    stage = icml.runner.STAGES["qa-gap-fill-synthesis"]
    exit_code, summary = icml.runner.run_qa_gap_fill_synthesis(
        icml.args("qa-gap-fill-synthesis", "--provider-cost-approved"), stage, llm_client=client
    )
    assert exit_code == 0 and summary["status"] == "succeeded", summary["stages"][0].get("error")
    return client


def test_icml_qa_gap_fill_synthesis_success_receipt(icml: IcmlHarness) -> None:
    run_through(icml, "analysis")
    client = run_stage8_success(icml)

    assert client.calls == ["executive_summary", "trend_report"]
    snapshot = _snap(
        icml,
        "qa-gap-fill-synthesis",
        mask=STAGE8_MASK,
        extra_output_paths=["analysis/qa", "analysis/executive_summary.md", "analysis/trend_report.md"],
    )
    print(json.dumps(snapshot["receipt"]["checks"], indent=1)[:3000])
    assert_golden(CONFERENCE, "stage08_qa_gap_fill_synthesis__succeeded_fake_client", snapshot)


def test_icml_qa_gap_fill_synthesis_resume_accepts_receipt_without_spend(icml: IcmlHarness) -> None:
    run_through(icml, "qa-gap-fill-synthesis")
    before = icml.receipt_file("qa-gap-fill-synthesis").read_bytes()

    class MustNotSynthesize:
        provider = "none"
        model = "none"

        def synthesize(self, **_kwargs: Any) -> Any:
            raise AssertionError("resume re-ran paid synthesis")

    stage = icml.runner.STAGES["qa-gap-fill-synthesis"]
    exit_code, summary = icml.runner.run_qa_gap_fill_synthesis(
        icml.args("qa-gap-fill-synthesis", "--resume", "--provider-cost-approved"), stage, llm_client=MustNotSynthesize()
    )

    assert (exit_code, summary["status"]) == (0, "succeeded")
    assert icml.receipt_file("qa-gap-fill-synthesis").read_bytes() == before


def test_icml_qa_gap_fill_synthesis_blocks_without_stage7(icml: IcmlHarness) -> None:
    run_through(icml, "exhaustive-topic-extraction")
    exit_code, summary = icml.runner.run_qa_gap_fill_synthesis(
        icml.args("qa-gap-fill-synthesis", "--provider-cost-approved"),
        icml.runner.STAGES["qa-gap-fill-synthesis"],
        llm_client=FakeStage8Client(icml.runner),
    )

    assert (exit_code, summary["status"]) == (1, "blocked")
    assert_golden(
        CONFERENCE,
        "stage08_qa_gap_fill_synthesis__blocked_missing_stage7",
        _snap(icml, "qa-gap-fill-synthesis", mask=STAGE8_MASK),
    )


# --------------------------------------------------------------------------
# Stage 9: podcast
# --------------------------------------------------------------------------


class FakePodcastCommands:
    """Stand-in for the script-generation and TTS subprocesses (no provider)."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> Any:
        import subprocess as _subprocess

        self.commands.append(list(command))
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        if "scripts/generate_focused_podcast.py" in command:
            script = {
                "title": "Fixture podcast",
                "summary": "A source-backed fixture podcast.",
                "segments": [
                    {"speaker": "Maya", "text": " ".join(["This is a source-backed fixture segment."] * 30)},
                    {"speaker": "Leo", "text": " ".join(["This segment continues the fixture analysis."] * 30)},
                ],
                "show_notes": "Generated from fixture reports.",
            }
            write_json(output_dir / "script.json", script)
            (output_dir / "script.md").write_text("# Fixture podcast\n", encoding="utf-8")
            (output_dir / "show_notes.md").write_text("# Show notes\n", encoding="utf-8")
            return _subprocess.CompletedProcess(command, 0, "script ok", "")
        if "scripts/render_podcast_audio.py" in command:
            audio_path = output_dir / "podcast.mp3"
            audio_path.write_bytes(b"fixture-mp3-bytes")
            write_json(
                output_dir / "audio_manifest.json",
                {
                    "schema_version": 1,
                    "podcast_path": str(audio_path),
                    "podcast_byte_count": audio_path.stat().st_size,
                    "segment_count": 2,
                    "audio_file_count": 2,
                    "secrets_copied": False,
                },
            )
            return _subprocess.CompletedProcess(command, 0, "audio ok", "")
        raise AssertionError(command)


FAKE_PODCAST_PROVIDER_SUMMARY = {
    "canonical_receipt_files_scanned": 2,
    "canonical_total_estimated_cost_usd": 0.02,
    "canonical_totals_by_provider": {"fixture-provider": 0.02},
    "canonical_totals_by_model": {"fixture-provider:fixture-tts": 0.02},
    "canonical_totals_by_operation": {"tts": 0.02},
    "failed_receipts": 0,
    "unknown_cost_receipts": 0,
    "receipt_paths": ["provider-call-1.json", "provider-call-2.json"],
}


def run_podcast_success(icml: IcmlHarness) -> FakePodcastCommands:
    commands = FakePodcastCommands()
    # The real script/TTS subprocesses write canonical provider receipts; the
    # fakes do not, so the summary over them is canned.
    icml.monkeypatch.setattr(
        icml.runner, "provider_receipt_summary", lambda *_a, **_k: dict(FAKE_PODCAST_PROVIDER_SUMMARY)
    )
    exit_code, summary = icml.runner.run_podcast(
        icml.args("podcast", "--provider-cost-approved"), icml.runner.STAGES["podcast"], command_runner=commands
    )
    assert (exit_code, summary["status"]) == (0, "succeeded"), summary["stages"][0].get("error")
    return commands


def test_icml_podcast_success_receipt(icml: IcmlHarness) -> None:
    run_through(icml, "qa-gap-fill-synthesis")
    commands = run_podcast_success(icml)

    assert [Path(command[1]).name for command in commands.commands] == [
        "generate_focused_podcast.py",
        "render_podcast_audio.py",
    ]
    assert_golden(
        CONFERENCE,
        "stage09_podcast__succeeded_fake_commands",
        _snap(icml, "podcast", extra_output_paths=["audio", "analysis/podcast_source.md"]),
    )


def test_icml_podcast_resume_accepts_receipt_without_rerun(icml: IcmlHarness) -> None:
    run_through(icml, "podcast")
    before = icml.receipt_file("podcast").read_bytes()

    def must_not_run(command: list[str]) -> Any:
        raise AssertionError(f"resume re-ran {command}")

    exit_code, summary = icml.runner.run_podcast(
        icml.args("podcast", "--resume", "--provider-cost-approved"), icml.runner.STAGES["podcast"], command_runner=must_not_run
    )

    assert (exit_code, summary["status"]) == (0, "succeeded")
    assert icml.receipt_file("podcast").read_bytes() == before


def test_icml_podcast_blocks_without_cost_approval(icml: IcmlHarness) -> None:
    run_through(icml, "qa-gap-fill-synthesis")

    def must_not_run(command: list[str]) -> Any:
        raise AssertionError(f"unapproved podcast ran {command}")

    exit_code, summary = icml.runner.run_podcast(
        icml.args("podcast"), icml.runner.STAGES["podcast"], command_runner=must_not_run
    )

    assert (exit_code, summary["status"]) == (1, "blocked")
    assert_golden(CONFERENCE, "stage09_podcast__blocked_no_cost_approval", _snap(icml, "podcast"))


def test_icml_podcast_blocks_when_stage8_outputs_did_not_pass(icml: IcmlHarness) -> None:
    """W5 Phase B: ICML now gates Stage 9 on the Stage 8 receipt's
    executive-summary / trend-report statuses, as CVPR always did."""

    run_through(icml, "qa-gap-fill-synthesis")
    stage8_path = icml.receipt_file("qa-gap-fill-synthesis")
    stage8 = read_json(stage8_path)
    assert (stage8["checks"]["executive_summary_status"], stage8["checks"]["trend_report_status"]) == ("pass", "pass")
    stage8["checks"]["trend_report_status"] = "fail"
    write_json(stage8_path, stage8)

    def must_not_run(command: list[str]) -> Any:
        raise AssertionError(f"podcast ran on a failed Stage 8 output: {command}")

    exit_code, summary = icml.runner.run_podcast(
        icml.args("podcast", "--provider-cost-approved"), icml.runner.STAGES["podcast"], command_runner=must_not_run
    )

    assert (exit_code, summary["status"]) == (1, "blocked")
    assert "podcast_stage8_trend_report_status_not_pass" in json.dumps(summary["stages"][0]["checks"])


# --------------------------------------------------------------------------
# Resume: Stage 7A accepts its own persisted receipt without re-extracting
# --------------------------------------------------------------------------


def test_icml_exhaustive_topic_extraction_resume_accepts_receipt_without_rerun(icml: IcmlHarness) -> None:
    run_through(icml, "exhaustive-topic-extraction")
    before = stable_snapshot(icml.receipt("exhaustive-topic-extraction"), workspace=icml.workspace)
    icml.monkeypatch.setattr(
        icml.runner, "run_exhaustive_topic_extraction", _must_not_rebuild
    )

    exit_code, summary = icml.run("exhaustive-topic-extraction", "--resume")

    assert (exit_code, summary["status"]) == (0, "succeeded")
    after = stable_snapshot(icml.receipt("exhaustive-topic-extraction"), workspace=icml.workspace)
    assert after == before


# --------------------------------------------------------------------------
# Lineage invalidation: a changed upstream output blocks before any spend
# --------------------------------------------------------------------------


def test_icml_podcast_blocks_before_spend_when_a_stage8_output_was_edited(icml: IcmlHarness) -> None:
    """An out-of-band edit to a Stage 8 output (executive_summary.md) blocks
    the paid podcast before any command runs: the pre-spend gate re-digests
    the upstream stage's recorded output artifacts, not just its receipt
    chain, and stage-status reports Stage 8 itself as stale."""

    from conference_pipeline.receipt_lineage import stage_lineage_report

    run_through(icml, "qa-gap-fill-synthesis")
    summary_path = icml.workspace / "analysis" / "executive_summary.md"
    summary_path.write_text(summary_path.read_text(encoding="utf-8") + "\nOut-of-band edit.\n", encoding="utf-8")

    def must_not_run(command: list[str]) -> Any:
        raise AssertionError(f"edited-upstream-output podcast ran {command}")

    exit_code, summary = icml.runner.run_podcast(
        icml.args("podcast", "--provider-cost-approved"), icml.runner.STAGES["podcast"], command_runner=must_not_run
    )

    assert (exit_code, summary["status"]) == (1, "blocked")
    checks = summary["stages"][0]["checks"]
    assert checks["blocker_code"] == icml.runner.UPSTREAM_LINEAGE_NOT_FRESH
    assert checks["provider_calls_made"] is False
    [blocker] = checks["upstream_lineage_blockers"]
    assert blocker["code"] == "stage_contract_upstream_output_changed"
    assert blocker["upstream_stage_id"] == "qa-gap-fill-synthesis"
    assert blocker["paths"] == ["analysis/executive_summary.md"]
    assert_golden(
        CONFERENCE,
        "stage09_podcast__blocked_edited_stage8_output",
        _snap(icml, "podcast"),
    )

    report = stage_lineage_report(icml.workspace, RUN_ID, layout="production")
    stage8 = report["stages"]["qa-gap-fill-synthesis"]
    assert stage8["state"] == "stale"
    assert stage8["reasons"][0]["code"] == "stage_contract_output_changed"
    assert stage8["reasons"][0]["paths"] == ["analysis/executive_summary.md"]


def test_icml_podcast_blocks_before_spend_when_stage8_is_itself_stale(icml: IcmlHarness) -> None:
    run_through(icml, "qa-gap-fill-synthesis")
    # Stage 7's output moves after Stage 8 bound it: Stage 8 is stale.
    manifest = icml.workspace / "analysis" / "analysis_citation_manifest.json"
    payload = read_json(manifest)
    payload["w1_out_of_band_edit"] = True
    write_json(manifest, payload)

    def must_not_run(command: list[str]) -> Any:
        raise AssertionError(f"stale-upstream podcast ran {command}")

    exit_code, summary = icml.runner.run_podcast(
        icml.args("podcast", "--provider-cost-approved"), icml.runner.STAGES["podcast"], command_runner=must_not_run
    )

    assert (exit_code, summary["status"]) == (1, "blocked")
    assert summary["stages"][0]["checks"]["blocker_code"] == icml.runner.UPSTREAM_LINEAGE_NOT_FRESH
    assert_golden(CONFERENCE, "stage09_podcast__blocked_stale_upstream_chain", _snap(icml, "podcast"))


# --------------------------------------------------------------------------
# Stage table and CLI surface (W3 must preserve every id, order and flag)
# --------------------------------------------------------------------------


def _capture_parser(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Any:
    import argparse

    captured: dict[str, Any] = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self: argparse.ArgumentParser, args: Any = None, namespace: Any = None) -> Any:
        captured.setdefault("parser", self)
        return original(self, args, namespace)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    runner.parse_args(["--stage", "runner-preflight"])
    return captured["parser"]


def cli_surface(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    parser = _capture_parser(runner, monkeypatch)
    options = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        default = action.default
        options[" ".join(action.option_strings)] = {
            "dest": action.dest,
            "default": str(default) if isinstance(default, Path) else default,
            "choices": sorted(action.choices) if action.choices is not None else None,
            "required": action.required,
            "nargs": action.nargs,
            "action": type(action).__name__,
        }
    return options


def test_icml_stage_table_and_cli_surface(icml: IcmlHarness) -> None:
    stages = {
        stage_id: {
            "order": stage.order,
            "receipt_filename": stage.receipt_filename,
            "implemented": stage.implemented,
            "next_action": stage.next_action,
        }
        for stage_id, stage in icml.runner.STAGES.items()
    }
    snapshot = {
        "stages": stages,
        "cli": cli_surface(icml.runner, icml.monkeypatch),
        "receipt_type": icml.runner.STAGE_RECEIPT_TYPE,
        "code_identity": dict(icml.runner.RUNNER_CODE_IDENTITY),
    }
    from tests.orchestrator_characterization import snapshot_payload

    assert_golden(CONFERENCE, "runner_stage_table_and_cli", snapshot_payload(snapshot, workspace=icml.workspace))


def test_icml_download_corpus_resume_reuses_cached_artifacts(icml: IcmlHarness) -> None:
    run_through(icml, "download-corpus")
    first_calls = len(icml.web.calls)
    exit_code, summary = icml.run("download-corpus", "--resume")

    resumed_calls = icml.web.calls[first_calls:]
    assert (exit_code, summary["status"]) == (0, "succeeded")
    # Cached source pages, item pages, PDFs and media are reused; only the
    # OpenReview metadata query is re-issued.
    assert resumed_calls == [f"{icml.runner.OPENREVIEW_API_NOTES}?" + resumed_calls[0].split("?", 1)[1]]
    assert_golden(
        CONFERENCE,
        "stage04_download_corpus__resume_reused",
        _snap(icml, "download-corpus", extra_output_paths=[icml.runner.DOWNLOAD_CORPUS_OPENREVIEW]),
    )
