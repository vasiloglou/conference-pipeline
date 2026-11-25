"""Tests for the fail-closed secrets export gate."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from conference_pipeline.publishing import GateClosed, GateReceipt, no_secrets, write_gate_receipt
from conference_pipeline.publishing.export_gate import _gitleaks_content_scan, _path_scan


# ---------------------------------------------------------------------------
# Stub scanners (D-02: stub injection contract)
# ---------------------------------------------------------------------------

def _clean_stub(paths: list[Path]) -> list[dict]:
    return []


def _dirty_stub(paths: list[Path]) -> list[dict]:
    return [{"Description": "Generic API Key", "File": str(paths[0]) if paths else "unknown", "StartLine": 1}]


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------

def test_gate_receipt_is_dataclass():
    receipt = GateReceipt(
        gate_id="secrets",
        status="passed",
        timestamp_utc="2026-06-26T00:00:00Z",
        findings=[],
    )
    d = dataclasses.asdict(receipt)
    assert d["gate_id"] == "secrets"
    assert d["status"] == "passed"
    assert d["findings"] == []
    json.dumps(d)  # must be JSON-serialisable


def test_no_secrets_passes_clean_paths(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("hello", encoding="utf-8")
    receipt = no_secrets([f], _scanner=_clean_stub)
    assert isinstance(receipt, GateReceipt)
    assert receipt.status == "passed"
    assert receipt.findings == []


def test_no_secrets_raises_on_secret_like_path(tmp_path: Path):
    bad_path = tmp_path / "credentials.json"
    bad_path.write_text("{}", encoding="utf-8")
    with pytest.raises(GateClosed) as exc_info:
        no_secrets([bad_path], _scanner=_clean_stub)
    assert exc_info.value.reason in {"secret_like_path", "gitleaks_not_installed"}


def test_no_secrets_passes_empty_list():
    receipt = no_secrets([], _scanner=_clean_stub)
    assert receipt.status == "passed"
    assert receipt.findings == []


def test_no_secrets_path_scan_fires_before_content_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad_path = tmp_path / ".env"
    bad_path.write_text("SECRET=oops", encoding="utf-8")
    scanner_called = []

    def recording_stub(paths):
        scanner_called.append(True)
        return []

    with pytest.raises(GateClosed) as exc_info:
        no_secrets([bad_path], _scanner=recording_stub)
    assert not scanner_called, "path scan should abort before content scan"
    assert exc_info.value.reason == "secret_like_path"


def test_path_scan_detects_env_file(tmp_path: Path):
    findings = _path_scan([tmp_path / ".env"])
    assert len(findings) == 1
    assert findings[0]["reason"] == "secret_like_filename"


def test_path_scan_allows_normal_file(tmp_path: Path):
    findings = _path_scan([tmp_path / "readme.md"])
    assert findings == []


def test_gate_receipt_serialisable_via_asdict():
    receipt = GateReceipt(
        gate_id="secrets",
        status="failed",
        timestamp_utc="2026-06-26T00:00:00Z",
        findings=[{"path": "x.txt", "reason": "secret_like_filename"}],
        scanner_version="8.30.1",
    )
    d = dataclasses.asdict(receipt)
    assert "findings" in d
    assert d["scanner_version"] == "8.30.1"


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------

def test_no_secrets_fail_closed_when_gitleaks_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    safe_file = tmp_path / "somefile.txt"
    safe_file.write_text("content", encoding="utf-8")
    with pytest.raises(GateClosed) as exc_info:
        no_secrets([safe_file])
    assert "gitleaks" in exc_info.value.reason


def test_no_secrets_passes_with_clean_stub(tmp_path: Path):
    f = tmp_path / "data.txt"
    f.write_text("clean content", encoding="utf-8")
    receipt = no_secrets([f], _scanner=_clean_stub)
    assert receipt.status == "passed"
    assert receipt.findings == []


def test_no_secrets_fails_with_dirty_stub(tmp_path: Path):
    f = tmp_path / "data.txt"
    f.write_text("normal content", encoding="utf-8")
    with pytest.raises(GateClosed) as exc_info:
        no_secrets([f], _scanner=_dirty_stub)
    assert exc_info.value.receipt["status"] == "failed"
    assert len(exc_info.value.receipt["findings"]) > 0


def test_gitleaks_content_scan_preserves_duplicate_basenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first = tmp_path / "a" / "analysis.md"
    second = tmp_path / "b" / "analysis.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    copied_names: list[str] = []

    monkeypatch.setattr(shutil, "which", lambda _name: "gitleaks")

    def fake_run(cmd, **_kwargs):
        source_dir = Path(cmd[cmd.index("--source") + 1])
        report_path = Path(cmd[cmd.index("--report-path") + 1])
        copied_names.extend(sorted(path.name for path in source_dir.iterdir()))
        report_path.write_text("[]", encoding="utf-8")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _gitleaks_content_scan([first, second]) == []
    assert copied_names == ["000000_analysis.md", "000001_analysis.md"]


def test_gate_receipt_written_to_path(tmp_path: Path):
    f = tmp_path / "clean.txt"
    f.write_text("nothing suspicious", encoding="utf-8")
    receipt_path = tmp_path / "r.json"
    no_secrets([f], receipt_path=receipt_path, _scanner=_clean_stub)
    assert receipt_path.exists()
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert "status" in data
    assert data["status"] == "passed"


def test_gate_receipt_written_on_dirty_stub_failure(tmp_path: Path):
    f = tmp_path / "data.txt"
    f.write_text("content", encoding="utf-8")
    receipt_path = tmp_path / "fail_receipt.json"
    with pytest.raises(GateClosed):
        no_secrets([f], receipt_path=receipt_path, _scanner=_dirty_stub)
    assert receipt_path.exists()
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["status"] == "failed"


def test_write_gate_receipt_creates_parent_dirs(tmp_path: Path):
    receipt = GateReceipt(
        gate_id="secrets",
        status="passed",
        timestamp_utc="2026-06-26T00:00:00Z",
        findings=[],
    )
    deep_path = tmp_path / "a" / "b" / "c" / "receipt.json"
    write_gate_receipt(receipt, deep_path)
    assert deep_path.exists()
    data = json.loads(deep_path.read_text(encoding="utf-8"))
    assert data["status"] == "passed"


def test_a_digest_in_a_generated_site_manifest_is_not_a_secret(tmp_path: Path) -> None:
    """6b, CVPR 2026: a transcript named for YouTube id ...APIY made gitleaks' generic-api-key
    rule read its sha256 in site_manifest.json as a key. Digests in the generated manifests are
    not secrets; anything else there still is."""
    import shutil

    from conference_pipeline.publishing import export_gate

    if shutil.which("gitleaks") is None:
        pytest.skip("gitleaks not installed")
    import base64
    import hashlib

    transcript = tmp_path / "site" / "files" / "transcripts" / "video-ktcEBizAPIY.md"
    transcript.parent.mkdir(parents=True)
    transcript.write_bytes(b"a transcript")
    digest = hashlib.sha256(b"a transcript").hexdigest()  # a real, high-entropy digest
    clean = tmp_path / "site" / "site_manifest.json"
    clean.write_text('{\n  "files/transcripts/video-ktcEBizAPIY.md": "' + digest + '"\n}\n', encoding="utf-8")
    # A digest is exempt because it is the sha256 of a file the same scan reads.
    assert export_gate._gitleaks_content_scan([clean, transcript]) == []
    dirty = tmp_path / "other" / "site_manifest.json"
    dirty.parent.mkdir(parents=True)
    token = base64.b64encode(hashlib.sha256(b"not a digest").digest()).decode().rstrip("=")
    dirty.write_text('{\n  "api_key": "' + token + '"\n}\n', encoding="utf-8")
    assert export_gate._gitleaks_content_scan([dirty]) != []


def test_r1_f3_a_digest_shaped_secret_is_not_exempt_by_its_spelling(tmp_path: Path) -> None:
    """Review phase-6b-r1 F3: 64 hex characters in a file named like a generated manifest were
    discarded whatever they were. Only the sha256 of a file the same scan reads is exempt."""
    import hashlib
    import shutil

    from conference_pipeline.publishing import export_gate

    if shutil.which("gitleaks") is None:
        pytest.skip("gitleaks not installed")
    key = hashlib.sha256(b"a key, not any file's digest").hexdigest()
    manifest = tmp_path / "site" / "site_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{\n  "api_key": "' + key + '"\n}\n', encoding="utf-8")
    assert export_gate._gitleaks_content_scan([manifest]) != []
    # The same spelling as the sha256 of a scanned file is a checksum entry.
    other = tmp_path / "site" / "files" / "a.md"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"a key, not any file's digest")
    assert export_gate._gitleaks_content_scan([manifest, other]) == []


@pytest.mark.parametrize("report", [None, "not json", '{"a": 1}', "[]"], ids=["missing", "malformed", "not-a-list", "empty"])
def test_r3_f3_a_failed_gitleaks_run_closes_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report) -> None:
    """Review phase-6b-r3 F3: the scanner's exit code was ignored and an unreadable report
    read as no findings, so a failed scan passed the gate."""
    import subprocess

    from conference_pipeline.publishing import export_gate

    target = tmp_path / "a.md"
    target.write_text("text", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda _name: "gitleaks")

    def fake_run(argv, **_kw):
        if report is not None:
            Path(argv[argv.index("--report-path") + 1]).write_text(report, encoding="utf-8")
        return subprocess.CompletedProcess(argv, 2, "", "error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GateClosed):
        no_secrets([target])
    # A run that exits 0 but leaves no readable list of findings is no scan either.
    if report != "[]":
        monkeypatch.setattr(subprocess, "run", lambda argv, **kw: (fake_run(argv, **kw), subprocess.CompletedProcess(argv, 0, "", ""))[1])
        with pytest.raises(GateClosed):
            no_secrets([target])
    assert export_gate is not None


def test_r3_f9_the_digest_exemption_holds_for_the_generated_manifests_only(tmp_path: Path) -> None:
    import hashlib

    from conference_pipeline.publishing import export_gate

    if shutil.which("gitleaks") is None:
        pytest.skip("gitleaks not installed")
    other = tmp_path / "site" / "files" / "a.md"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"a key, not any file's digest")
    key = hashlib.sha256(other.read_bytes()).hexdigest()
    notes = tmp_path / "site" / "notes_site_manifest.json"
    notes.write_text('{\n  "api_key": "' + key + '"\n}\n', encoding="utf-8")
    assert export_gate._gitleaks_content_scan([notes, other]) != []
    manifest = tmp_path / "site" / "site_manifest.json"
    manifest.write_text('{\n  "api_key": "' + key + '"\n}\n', encoding="utf-8")
    assert export_gate._gitleaks_content_scan([manifest, other]) == []
