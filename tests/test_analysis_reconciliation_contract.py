from __future__ import annotations
import pytest

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_DIR = (
    REPO_ROOT
    / ".planning"
    / "milestones"
    / "v1.3-phases"
    / "29-analysis-reconciliation-verification-and-drift-guards"
)
SPEC_PATH = PHASE_DIR / "29-analysis-reconciliation-spec.json"
REPORT_PATH = PHASE_DIR / "29-analysis-reconciliation-report.json"
PROOF_PATH = PHASE_DIR / "29-analysis-reconciliation-proof.md"
CHECKER = REPO_ROOT / "scripts" / "check_analysis_reconciliation.py"
SECRET_PATTERNS = (
    "sk-",
    "AIza",
    "ANTHROPIC_API_KEY=",
    "OPENAI_API_KEY=",
    "GOOGLE_API_KEY=",
    "GEMINI_API_KEY=",
    "TAVILY_API_KEY=",
    "Cookie:",
    "NEURIPS_PASSWORD=",
)
REQUIRED_ITEM_IDS = {
    "workflow-output-command",
    "workflow-resume-command",
    "workflow-summary-command",
    "conference-smoketest-command",
    "analysis-manifest-contract",
    "vector-store-enhanced-script",
    "vector-store-legacy-script",
    "vector-store-cache-path",
    "gpt-researcher-vectorstore-script",
    "gpt-researcher-hybrid-query-script",
    "deep-research-export-script",
    "qa-analysis-script",
    "traceability-recovery-script",
    "inline-traceability-script",
    "focused-podcast-workflow-script",
    "rank-facts-local-script",
    "slides-agentic-local-script",
    "podcast-gemini-historical-script",
    "phase8-paper-analysis-script",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_spec() -> dict[str, object]:
    return json.loads(read_text(SPEC_PATH))


def run_checker(spec: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--spec", str(spec), "--output", str(output)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_reconciliation_checker_rejects_malformed_spec(tmp_path: Path) -> None:
    spec = tmp_path / "bad.json"
    output = tmp_path / "report.json"
    spec.write_text("{not-json", encoding="utf-8")

    result = run_checker(spec, output)

    report = json.loads(read_text(output))
    assert result.returncode != 0
    assert report["status"] == "failed"
    assert "malformed spec" in json.dumps(report["failures"])


def test_reconciliation_checker_rejects_repo_escape_paths(tmp_path: Path) -> None:
    spec = tmp_path / "escape.json"
    output = tmp_path / "report.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "items": [
                    {
                        "id": "escape",
                        "kind": "script",
                        "path": "../outside.py",
                        "expected_docs": ["../outside.md"],
                        "expected_phrases": [],
                        "expected_labels": [],
                        "safe_checks": ["exists"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_checker(spec, output)

    report = json.loads(read_text(output))
    assert result.returncode != 0
    assert report["status"] == "failed"
    assert "path escapes repository" in json.dumps(report["failures"])


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/29-analysis-reconciliation-verification-and-drift-guards/29-analysis-reconciliation-spec.json")
def test_phase_spec_contains_required_reconciliation_items() -> None:
    spec = load_spec()
    item_ids = {item["id"] for item in spec["items"]}

    assert spec["schema_version"] == 1
    assert REQUIRED_ITEM_IDS <= item_ids
    assert any("expected_labels" in item for item in spec["items"])
    assert any("limited_reason" in item for item in spec["items"])


@pytest.mark.repo_data(".planning/milestones/v1.3-phases")
def test_reconciliation_checker_report_passes_for_phase_spec(tmp_path: Path) -> None:
    # The checker writes to tmp_path: the tracked v1.3 report is the reference and
    # is never rewritten by the suite (v6.0 Phase 2, O34).
    output = tmp_path / "report.json"
    result = run_checker(SPEC_PATH, output)

    report = json.loads(read_text(output))
    assert result.returncode == 0, report["failures"]
    assert report["schema_version"] == 1
    assert report["status"] == "passed"
    assert set(report["checked_items"]) >= REQUIRED_ITEM_IDS
    assert report["failures"] == []
    tracked = json.loads(read_text(REPORT_PATH))
    assert sorted(report["checked_items"]) == sorted(tracked["checked_items"])


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/29-analysis-reconciliation-verification-and-drift-guards/29-analysis-reconciliation-spec.json")
def test_reconciliation_checker_fails_when_doc_phrase_missing(tmp_path: Path) -> None:
    spec = load_spec()
    spec["items"][0]["expected_phrases"] = ["phrase that is intentionally absent"]
    spec_path = tmp_path / "missing-phrase.json"
    output = tmp_path / "report.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = run_checker(spec_path, output)

    report = json.loads(read_text(output))
    assert result.returncode != 0
    assert "missing expected phrase" in json.dumps(report["failures"])


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/29-analysis-reconciliation-verification-and-drift-guards/29-analysis-reconciliation-spec.json")
def test_reconciliation_checker_fails_when_required_script_missing(tmp_path: Path) -> None:
    spec = load_spec()
    item = next(item for item in spec["items"] if item["id"] == "qa-analysis-script")
    item["path"] = "scripts/not_a_real_analysis_script.py"
    item["expected_phrases"] = []
    spec_path = tmp_path / "missing-script.json"
    output = tmp_path / "report.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = run_checker(spec_path, output)

    report = json.loads(read_text(output))
    assert result.returncode != 0
    assert "missing required path" in json.dumps(report["failures"])


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/29-analysis-reconciliation-verification-and-drift-guards/29-analysis-reconciliation-spec.json")
def test_reconciliation_checker_fails_when_expected_label_missing(tmp_path: Path) -> None:
    spec = load_spec()
    item = next(item for item in spec["items"] if item["id"] == "workflow-output-command")
    item["expected_labels"] = ["unsupported"]
    item["expected_docs"] = ["docs/source-of-truth.md"]
    item["expected_phrases"] = []
    spec_path = tmp_path / "missing-label.json"
    output = tmp_path / "report.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = run_checker(spec_path, output)

    report = json.loads(read_text(output))
    assert result.returncode != 0
    assert "missing expected_label" in json.dumps(report["failures"])


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/29-analysis-reconciliation-verification-and-drift-guards/29-analysis-reconciliation-spec.json")
def test_reconciliation_checker_fails_when_doc_script_reference_missing_from_spec(
    tmp_path: Path,
) -> None:
    spec = load_spec()
    spec["items"] = [
        item
        for item in spec["items"]
        if item["id"] != "traceability-recovery-script"
    ]
    spec_path = tmp_path / "missing-coverage.json"
    output = tmp_path / "report.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = run_checker(spec_path, output)

    report = json.loads(read_text(output))
    assert result.returncode != 0
    assert "missing script reference coverage" in json.dumps(report["failures"])
    assert "scripts/traceability_recovery.py" in json.dumps(report["failures"])


@pytest.mark.repo_data(".planning/milestones/v1.3-phases")
def test_reconciliation_checker_blocks_final_pass_without_phase28_evidence() -> None:
    summary = PHASE_DIR.parent / "28-analysis-workflow-contract-and-script-alignment" / "28-01-SUMMARY.md"
    assert summary.exists()
    assert "Phase 28" in read_text(summary)
    assert "phase28_completion_precondition" in read_text(CHECKER)


@pytest.mark.repo_data(".planning/milestones/v1.3-phases")
def test_reconciliation_checker_records_provider_heavy_limited_checks(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    run_checker(SPEC_PATH, output)
    report = json.loads(read_text(output))
    limited = {entry["path"]: entry["reason"] for entry in report["limited_checks"]}

    assert limited["scripts/rank_facts.py"] == (
        "provider-heavy help/import path intentionally not exercised"
    )
    assert limited["scripts/generate_focused_podcast.py"] == (
        "provider-heavy help/import path intentionally not exercised"
    )
    assert limited["scripts/generate_podcast_gemini.py"] == (
        "provider-heavy help/import path intentionally not exercised"
    )


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/29-analysis-reconciliation-verification-and-drift-guards/29-analysis-reconciliation-report.json")
def test_reconciliation_report_and_proof_do_not_leak_secret_patterns(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    run_checker(SPEC_PATH, output)
    combined = read_text(output) + read_text(REPORT_PATH) + read_text(PROOF_PATH)

    for pattern in SECRET_PATTERNS:
        assert pattern not in combined


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/29-analysis-reconciliation-verification-and-drift-guards/29-analysis-reconciliation-proof.md")
def test_reconciliation_proof_records_reproducible_commands() -> None:
    proof = read_text(PROOF_PATH)

    assert "Status: passed" in proof
    assert "29-analysis-reconciliation-report.json" in proof
    assert (
        ".venv/bin/python scripts/check_analysis_reconciliation.py --spec "
        ".planning/phases/29-analysis-reconciliation-verification-and-drift-guards/"
        "29-analysis-reconciliation-spec.json --output "
        ".planning/phases/29-analysis-reconciliation-verification-and-drift-guards/"
        "29-analysis-reconciliation-report.json"
    ) in proof
    assert ".venv/bin/python -m pytest tests/test_analysis_reconciliation_contract.py -x" in proof
    assert "scripts/rank_facts.py" in proof
    assert "provider-heavy help/import path intentionally not exercised" in proof
