from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_corpus_source_metadata import (  # noqa: E402
    COVERAGE_FLOORS,
    STRUCTURALLY_AUTHORLESS,
    evaluate_floors,
    main,
    measure_workspace_coverage,
    write_coverage_receipt,
)

from conference_pipeline.indexing.paper_source_metadata import (  # noqa: E402
    SOURCE_FAMILY_CLASS_ARTIFACT,
    SOURCE_FAMILY_CLASS_MAIN_PAPER,
    SOURCE_FAMILY_CLASS_WORKSHOP_PAPER,
)

_ALL_FAMILIES = {SOURCE_FAMILY_CLASS_MAIN_PAPER, SOURCE_FAMILY_CLASS_WORKSHOP_PAPER, SOURCE_FAMILY_CLASS_ARTIFACT}


# ---------------------------------------------------------------------------
# Fixture workspace builder
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _corpus_record(index: int, family_id: str, *, title: bool, authors: bool, provenance: str) -> dict:
    record = {"artifact_id": f"{family_id}:{index}", "source_family_id": family_id}
    if title:
        record["paper_title"] = f"Paper Title {family_id} {index}"
    if authors:
        record["authors"] = ["Alice Author", "Bob Author"]
    record["source_metadata_provenance"] = {
        "title": provenance if title else "unresolved",
        "authors": provenance if authors else "unresolved",
    }
    return record


def build_workspace(
    base_dir: Path,
    *,
    main_count: int = 100,
    main_authors_missing: int = 0,
    main_titles_missing: int = 0,
    workshop_count: int = 100,
    workshop_authors_missing: int = 0,
    workshop_titles_missing: int = 0,
    artifact_count: int = 20,
    artifact_titles_present: int = 0,
    run_id: str = "test-run",
    include_chunks: bool = True,
    include_topic_ledgers: bool = True,
    include_speaker_ledgers: bool = True,
) -> Path:
    """Build a minimal, realistically-shaped conference workspace fixture.

    corpus_inventory.json is the sole driver of the tallied denominators in
    these tests: the vector-store chunk store, topic reference ledgers, and
    speaker/author work ledgers are created (so their absence never spuriously
    triggers a ``ledgers_missing``/``vector_store_missing`` finding) but empty,
    keeping the coverage-percentage arithmetic exact and easy to reason about.
    """

    workspace = base_dir / "workspace"

    records: list[dict] = []
    for i in range(main_count):
        records.append(
            _corpus_record(
                i,
                "paper_text",
                title=i >= main_titles_missing,
                authors=i >= main_authors_missing,
                provenance="source_discovery_join",
            )
        )
    for i in range(workshop_count):
        records.append(
            _corpus_record(
                i,
                "workshop_paper_text",
                title=i >= workshop_titles_missing,
                authors=i >= workshop_authors_missing,
                provenance="workshop_receipt",
            )
        )
    for i in range(artifact_count):
        records.append(
            _corpus_record(
                i,
                "speech_transcript",
                title=i < artifact_titles_present,
                authors=False,
                provenance="unresolved",
            )
        )
    _write_json(workspace / "storage" / "corpus" / "corpus_inventory.json", {"artifact_records": records})

    if include_chunks:
        chunks_path = workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.write_text("", encoding="utf-8")
    if include_topic_ledgers:
        _write_json(workspace / "analysis" / "topic_reference_ledgers" / "01_test_topic.json", {"entries": []})
    if include_speaker_ledgers:
        _write_json(workspace / "decks" / "01_test_topic" / "speaker_author_work_ledger.json", {"entries": []})

    return workspace


def _write_config_yaml(workspace: Path, floor_overrides: dict) -> None:
    config = {"corpus_metadata_coverage": {"floor_overrides": floor_overrides}}
    (workspace / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _measure_and_evaluate(workspace: Path, run_id: str = "test-run") -> tuple[dict, list[dict]]:
    measurements = measure_workspace_coverage(workspace, run_id=run_id)
    findings = evaluate_floors(measurements)
    return measurements, findings


# ---------------------------------------------------------------------------
# Behavior: full-coverage fixture passes with a per-family PASS report
# ---------------------------------------------------------------------------


def test_full_coverage_fixture_passes_with_per_family_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    workspace = build_workspace(tmp_path, artifact_titles_present=12)
    rc = main(["--workspace", str(workspace), "--run-id", "test-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    for family in _ALL_FAMILIES:
        assert family in out


# ---------------------------------------------------------------------------
# Behavior: below-floor main-author coverage fails and names the specifics
# ---------------------------------------------------------------------------


def test_below_floor_main_author_coverage_fails_and_names_details(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    workspace = build_workspace(tmp_path, main_authors_missing=10, artifact_titles_present=12)
    rc = main(["--workspace", str(workspace), "--run-id", "test-run"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert SOURCE_FAMILY_CLASS_MAIN_PAPER in out
    assert "0.99" in out  # the floor
    assert "0.9" in out  # the measured value (0.90) appears somewhere in the report


# ---------------------------------------------------------------------------
# Behavior: exactly three family sections, never a blended number
# ---------------------------------------------------------------------------


def test_report_has_exactly_three_families_never_blended(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    workspace = build_workspace(tmp_path, artifact_titles_present=12)
    rc = main(["--workspace", str(workspace), "--run-id", "test-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["families"].keys()) == _ALL_FAMILIES
    assert len(payload["families"]) == 3
    forbidden_terms = ("blended", "aggregate", "overall", "combined_total")
    for key in payload["families"]:
        assert not any(term in key.lower() for term in forbidden_terms)
    assert not any(term in payload for term in forbidden_terms)


# ---------------------------------------------------------------------------
# Behavior: artifacts family is reported, structurally_authorless, non-gating
# ---------------------------------------------------------------------------


def test_artifacts_family_reported_structurally_authorless_and_not_gated(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, artifact_count=50, artifact_titles_present=1)
    measurements, findings = _measure_and_evaluate(workspace)
    artifacts_report = measurements["families"][SOURCE_FAMILY_CLASS_ARTIFACT]
    assert artifacts_report["authors_classification"] == STRUCTURALLY_AUTHORLESS
    assert artifacts_report["total"] == 50
    assert artifacts_report["title_present"] == 1
    assert artifacts_report["authors_present"] == 0
    assert not any(
        f["code"] == "coverage_below_floor" and f.get("family") == SOURCE_FAMILY_CLASS_ARTIFACT for f in findings
    )


# ---------------------------------------------------------------------------
# Behavior: missing ledgers -> distinct non-zero exit, not spurious pass/crash
# ---------------------------------------------------------------------------


def test_missing_ledgers_yields_ledgers_missing_finding_not_spurious_pass(tmp_path: Path) -> None:
    workspace = build_workspace(
        tmp_path,
        artifact_titles_present=12,
        include_topic_ledgers=False,
        include_speaker_ledgers=False,
    )
    measurements, findings = _measure_and_evaluate(workspace)
    codes = {f["code"] for f in findings}
    assert "ledgers_missing" in codes
    # Coverage that IS measurable (from corpus_inventory) must still be
    # reported honestly, not smashed to a spurious 0%.
    assert measurements["families"][SOURCE_FAMILY_CLASS_MAIN_PAPER]["title_coverage_pct"] == 100.0

    rc = main(["--workspace", str(workspace), "--run-id", "test-run"])
    assert rc == 1


# ---------------------------------------------------------------------------
# Behavior: --json matches the human report's numbers
# ---------------------------------------------------------------------------


def test_json_output_matches_measured_numbers(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    workspace = build_workspace(tmp_path, artifact_titles_present=12)
    measurements, _ = _measure_and_evaluate(workspace)
    rc = main(["--workspace", str(workspace), "--run-id", "test-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["families"][SOURCE_FAMILY_CLASS_MAIN_PAPER]["title_coverage_pct"] == (
        measurements["families"][SOURCE_FAMILY_CLASS_MAIN_PAPER]["title_coverage_pct"]
    )
    assert payload["families"][SOURCE_FAMILY_CLASS_ARTIFACT]["title_present"] == 12


# ---------------------------------------------------------------------------
# Behavior: floor override -- unjustified refusal, justified acceptance
# ---------------------------------------------------------------------------


def test_floor_override_below_default_without_justification_is_refused(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, main_authors_missing=10, artifact_titles_present=12)
    _write_config_yaml(workspace, {SOURCE_FAMILY_CLASS_MAIN_PAPER: {"authors_floor": 0.5}})
    _, findings = _measure_and_evaluate(workspace)
    assert any(f["code"] == "floor_override_unjustified" for f in findings)

    rc = main(["--workspace", str(workspace), "--run-id", "test-run"])
    assert rc == 1


def test_floor_override_below_default_with_justification_is_honored_and_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    workspace = build_workspace(tmp_path, main_authors_missing=10, artifact_titles_present=12)
    _write_config_yaml(
        workspace,
        {SOURCE_FAMILY_CLASS_MAIN_PAPER: {"authors_floor": 0.85, "justification": "test override reason xyz"}},
    )
    rc = main(["--workspace", str(workspace), "--run-id", "test-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0.99" in out  # previous value printed
    assert "0.85" in out  # new value printed
    assert "test override reason xyz" in out

    measurements, findings = _measure_and_evaluate(workspace)
    assert not any(f["code"] == "floor_override_unjustified" for f in findings)
    override = measurements["effective_floors"][SOURCE_FAMILY_CLASS_MAIN_PAPER]["authors_floor_override"]
    assert override == {
        "previous_value": 0.99,
        "new_value": 0.85,
        "justification": "test override reason xyz",
    }


# ---------------------------------------------------------------------------
# Behavior: never writes to the corpus, vector store, or any ledger
# ---------------------------------------------------------------------------


def test_audit_never_mutates_measured_artifacts(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, main_count=5, workshop_count=5, artifact_count=2)
    corpus_inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    before_bytes = corpus_inventory_path.read_bytes()
    before_mtime = corpus_inventory_path.stat().st_mtime_ns

    measure_workspace_coverage(workspace, run_id="test-run")

    assert corpus_inventory_path.read_bytes() == before_bytes
    assert corpus_inventory_path.stat().st_mtime_ns == before_mtime


# ---------------------------------------------------------------------------
# Drop-and-fail proof: the gate is not tautological
# ---------------------------------------------------------------------------


def test_drop_and_fail_main_author_coverage_below_floor(tmp_path: Path) -> None:
    passing = build_workspace(tmp_path / "passing", artifact_titles_present=12)
    _, passing_findings = _measure_and_evaluate(passing)
    assert passing_findings == []

    degraded = build_workspace(tmp_path / "degraded", main_authors_missing=5, artifact_titles_present=12)
    degraded_measurements, degraded_findings = _measure_and_evaluate(degraded)
    assert degraded_measurements["families"][SOURCE_FAMILY_CLASS_MAIN_PAPER]["authors_coverage_pct"] == 95.0
    assert any(
        f["code"] == "coverage_below_floor" and f.get("family") == SOURCE_FAMILY_CLASS_MAIN_PAPER
        for f in degraded_findings
    )
    rc = main(["--workspace", str(degraded), "--run-id", "test-run"])
    assert rc == 1


# ---------------------------------------------------------------------------
# Streaming proof: the 70,000-line chunk store is never loaded whole
# ---------------------------------------------------------------------------


def test_chunk_store_streamed_not_loaded_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = build_workspace(
        tmp_path, main_count=1, workshop_count=1, artifact_count=0, include_chunks=False
    )
    chunks_path = workspace / "storage" / "vector-store" / "test-run" / "stores" / "papers" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as handle:
        for i in range(70_000):
            handle.write(
                json.dumps(
                    {
                        "source_context": {
                            "source_family_id": "paper_text",
                            "paper_title": f"Chunk Paper {i}",
                            "authors": ["Chunk Author"],
                        }
                    }
                )
                + "\n"
            )

    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == chunks_path:
            raise AssertionError("chunks.jsonl must be streamed, not loaded whole via Path.read_text")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    measurements = measure_workspace_coverage(workspace, run_id="test-run")
    # 1 corpus_inventory record + 70,000 chunk lines, all paper_text family.
    assert measurements["families"][SOURCE_FAMILY_CLASS_MAIN_PAPER]["total"] == 70_001


# ---------------------------------------------------------------------------
# classify_record_family reuse: topic-reference-ledger entries are correctly
# bridged to the exact family-id vocabulary, path-based, never blended.
# ---------------------------------------------------------------------------


def test_topic_reference_ledger_entries_classified_by_path_not_blended(tmp_path: Path) -> None:
    workspace = build_workspace(
        tmp_path,
        main_count=0,
        workshop_count=0,
        artifact_count=0,
        include_topic_ledgers=False,
    )
    _write_json(
        workspace / "analysis" / "topic_reference_ledgers" / "01_topic.json",
        {
            "entries": [
                {
                    "source_family": "papers",
                    "source_paths": ["storage/workshops/papers/x/y.pdf"],
                    "titles": ["Workshop Title"],
                    "metadata": {"authors": ["A"]},
                },
                {
                    "source_family": "papers",
                    "source_paths": ["storage/corpus/paper-text/x.txt"],
                    "titles": ["Main Title"],
                    "metadata": {"authors": ["B"]},
                },
                {
                    "source_family": "artifacts",
                    "source_paths": ["storage/download-corpus/event-media/captions/x.vtt"],
                    "titles": [],
                    "metadata": {},
                },
            ]
        },
    )
    measurements = measure_workspace_coverage(workspace, run_id="test-run")
    families = measurements["families"]
    assert families[SOURCE_FAMILY_CLASS_WORKSHOP_PAPER]["total"] == 1
    assert families[SOURCE_FAMILY_CLASS_MAIN_PAPER]["total"] == 1
    assert families[SOURCE_FAMILY_CLASS_ARTIFACT]["total"] == 1


# ---------------------------------------------------------------------------
# Every floor's justification cites an execution-time measurement
# ---------------------------------------------------------------------------


def test_every_coverage_floor_justification_cites_execution_time_measurement() -> None:
    for family_class, floor in COVERAGE_FLOORS.items():
        justification = floor["justification"]
        assert "2026-07-31" in justification, f"{family_class} justification missing execution date"
        assert re.search(r"\d+/\d+", justification) or re.search(r"\d+\.\d+%", justification), (
            f"{family_class} justification missing a numerator/denominator or percentage"
        )


# ---------------------------------------------------------------------------
# Task 2: coverage receipt
# ---------------------------------------------------------------------------

_REQUIRED_RECEIPT_KEYS = (
    "receipt_type",
    "status",
    "generated_at",
    "workspace",
    "run_id",
    "secrets_copied",
    "families",
    "floors",
    "findings",
    "inputs",
)


def test_receipt_contains_all_required_keys(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, artifact_titles_present=12)
    measurements, findings = _measure_and_evaluate(workspace)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-run" / "corpus_source_metadata_coverage.json"
    receipt = write_coverage_receipt(
        receipt_path,
        measurements=measurements,
        floors=measurements["effective_floors"],
        findings=findings,
        workspace=workspace,
        run_id="test-run",
        status="succeeded",
    )
    for key in _REQUIRED_RECEIPT_KEYS:
        assert key in receipt
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key in _REQUIRED_RECEIPT_KEYS:
        assert key in written


def test_receipt_secrets_copied_is_exactly_false(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, artifact_titles_present=12)
    measurements, findings = _measure_and_evaluate(workspace)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-run" / "corpus_source_metadata_coverage.json"
    receipt = write_coverage_receipt(
        receipt_path,
        measurements=measurements,
        floors=measurements["effective_floors"],
        findings=findings,
        workspace=workspace,
        run_id="test-run",
        status="succeeded",
    )
    assert receipt["secrets_copied"] is False


def test_failing_audit_still_writes_blocked_receipt(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, main_authors_missing=10, artifact_titles_present=12)
    rc = main(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "test-run",
            "--write-receipt",
            "storage/workflow-runs/test-run",
        ]
    )
    assert rc == 1
    receipt_path = workspace / "storage" / "workflow-runs" / "test-run" / "corpus_source_metadata_coverage.json"
    assert receipt_path.is_file()
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["status"] == "blocked"


def test_write_receipt_with_relative_workspace_does_not_double_nest_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 10.1-10 bug fix: a RELATIVE ``--workspace`` argument (the
    documented CLI usage, e.g. ``--workspace conferences/cvpr2026``) must not
    cause the receipt to land under a doubled workspace segment
    (``conferences/cvpr2026/conferences/cvpr2026/...``). This reproduces the
    real failure mode -- every other receipt-path test in this file passes an
    already-absolute ``tmp_path`` workspace, which never triggers the second
    join in ``write_coverage_receipt``, so this is the one regression test
    that actually exercises a relative ``--workspace``."""

    workspace = build_workspace(tmp_path, artifact_titles_present=12)
    monkeypatch.chdir(tmp_path)
    relative_workspace = workspace.name

    rc = main(
        [
            "--workspace",
            relative_workspace,
            "--run-id",
            "test-run",
            "--write-receipt",
            "storage/workflow-runs/test-run",
        ]
    )
    assert rc == 0
    correct_receipt_path = (
        workspace / "storage" / "workflow-runs" / "test-run" / "corpus_source_metadata_coverage.json"
    )
    assert correct_receipt_path.is_file()
    doubled_receipt_path = (
        workspace
        / relative_workspace
        / "storage"
        / "workflow-runs"
        / "test-run"
        / "corpus_source_metadata_coverage.json"
    )
    assert not doubled_receipt_path.exists()


def test_receipt_never_contains_fixture_author_names_or_titles(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, main_count=5, workshop_count=5, artifact_count=2)
    measurements, findings = _measure_and_evaluate(workspace)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-run" / "corpus_source_metadata_coverage.json"
    write_coverage_receipt(
        receipt_path,
        measurements=measurements,
        floors=measurements["effective_floors"],
        findings=findings,
        workspace=workspace,
        run_id="test-run",
        status="succeeded",
    )
    raw = receipt_path.read_bytes()
    assert b"Alice Author" not in raw
    assert b"Bob Author" not in raw
    assert b"Paper Title" not in raw


def test_receipt_path_outside_workspace_is_refused_no_write(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, main_count=5, workshop_count=5, artifact_count=2)
    outside = tmp_path / "outside" / "receipt.json"
    measurements, findings = _measure_and_evaluate(workspace)

    with pytest.raises(ValueError):
        write_coverage_receipt(
            outside,
            measurements=measurements,
            floors=measurements["effective_floors"],
            findings=findings,
            workspace=workspace,
            run_id="test-run",
            status="succeeded",
        )
    assert not outside.exists()

    rc = main(["--workspace", str(workspace), "--run-id", "test-run", "--write-receipt", str(outside)])
    assert rc == 1
    assert not outside.exists()


def test_receipt_inputs_map_to_64_char_hex_digests(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, main_count=5, workshop_count=5, artifact_count=2)
    measurements, findings = _measure_and_evaluate(workspace)
    receipt_path = workspace / "storage" / "workflow-runs" / "test-run" / "corpus_source_metadata_coverage.json"
    receipt = write_coverage_receipt(
        receipt_path,
        measurements=measurements,
        floors=measurements["effective_floors"],
        findings=findings,
        workspace=workspace,
        run_id="test-run",
        status="succeeded",
    )
    assert receipt["inputs"], "expected at least one measured input artifact to be digested"
    for digest in receipt["inputs"].values():
        assert len(digest) == 64
        int(digest, 16)  # raises ValueError if not valid hex


def test_receipt_write_is_atomic_no_leftover_temp_files(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path, main_count=5, workshop_count=5, artifact_count=2)
    measurements, findings = _measure_and_evaluate(workspace)
    receipt_dir = workspace / "storage" / "workflow-runs" / "test-run"
    receipt_path = receipt_dir / "corpus_source_metadata_coverage.json"
    write_coverage_receipt(
        receipt_path,
        measurements=measurements,
        floors=measurements["effective_floors"],
        findings=findings,
        workspace=workspace,
        run_id="test-run",
        status="succeeded",
    )
    leftover_temp_files = list(receipt_dir.glob("*.tmp"))
    assert leftover_temp_files == []
    assert receipt_path.is_file()


# --- v6.0-Public-Release Phase 5 (D4): --scope inventory ------------------------------

import audit_corpus_source_metadata as audit  # noqa: E402


def _inventory_only_workspace(tmp_path, *, authors=True):
    workspace = tmp_path / "invonly"
    corpus = workspace / "storage" / "corpus"
    corpus.mkdir(parents=True)
    records = [
        {"source_family_id": "paper_text", "paper_title": f"Main {i}", "authors": ["A"] if authors else []}
        for i in range(3)
    ] + [{"source_family_id": "workshop_paper_text", "paper_title": "W", "authors": ["B"]}]
    (corpus / "corpus_inventory.json").write_text(json.dumps({"artifact_records": records}), encoding="utf-8")
    return workspace


def test_scope_inventory_passes_without_vector_store_or_ledgers(tmp_path):
    workspace = _inventory_only_workspace(tmp_path)
    assert audit.main(["--workspace", str(workspace), "--scope", "inventory", "--json"]) == 0
    measurements = audit.measure_workspace_coverage(workspace, scope="inventory")
    assert measurements["not_measured"] == list(audit.NOT_MEASURED_IN_INVENTORY_SCOPE)
    assert measurements["findings"] == []


def test_scope_inventory_still_fails_below_a_floor(tmp_path):
    workspace = _inventory_only_workspace(tmp_path, authors=False)
    assert audit.main(["--workspace", str(workspace), "--scope", "inventory", "--json"]) == 1


def test_scope_all_is_unchanged_without_vector_store(tmp_path):
    workspace = _inventory_only_workspace(tmp_path)
    assert audit.main(["--workspace", str(workspace), "--json"]) == 1
