"""v6.0 Phase 7: the fresh-history code export (spec T1-T3, T7, T9's planted cases)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline.publishing.code_export import (
    CodeExportConfig,
    CodeExportError,
    build_code_export,
    check_code_export,
    selected_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "export_code_repo.py"
# A GitHub noreply alias, built at run time so no line of this file reads as an address.
NAME, EMAIL = "Example Maintainer", "12345+example" + "@" + "users.noreply.github.com"
needs_gitleaks = pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks is not installed")

MANIFEST = """schema_version = 1
[export]
include = ["src/", "scripts/", "docs/", "README.md", "code_export.toml", ".gitignore"]
exclude = ["src/private/"]
"""


def _git(root: Path, *args: str) -> str:
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "HOME": str(root)}
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True, env=env).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "src-repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "private").mkdir()
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    (root / "notes").mkdir()
    (root / "src" / "pkg" / "a.py").write_text("print('a')\n", encoding="utf-8")
    (root / "src" / "private" / "b.py").write_text("print('b')\n", encoding="utf-8")
    (root / "notes" / "plan.md").write_text("not exported\n", encoding="utf-8")
    (root / "docs" / "index.md").write_text("# Docs\n\n[readme](../README.md)\n", encoding="utf-8")
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    (root / "code_export.toml").write_text(MANIFEST, encoding="utf-8")
    (root / ".gitignore").write_text("ignored.txt\n.env\n", encoding="utf-8")
    shutil.copy(REPO_ROOT / "scripts" / "check_docs_links.py", root / "scripts" / "check_docs_links.py")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    # Present in the checkout but never tracked: none may reach the export (T2).
    (root / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    (root / ".env").write_text("OPENAI_API_KEY=not-a-real-key\n", encoding="utf-8")
    (root / "src" / "pkg" / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    return root


def _config(repo: Path, out: Path, **kw) -> CodeExportConfig:
    return CodeExportConfig(project_root=repo, source_commit="HEAD", out_dir=out, author_name=NAME, author_email=EMAIL, **kw)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)


EXPECTED = [".gitignore", "README.md", "code_export.toml", "docs/index.md", "scripts/check_docs_links.py", "src/pkg/a.py"]


def test_t1_dry_run_lists_the_selection_and_writes_nothing(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = _run("--project-root", str(repo), "--dry-run", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[: len(EXPECTED)] == EXPECTED
    assert "nothing written" in result.stdout
    assert not out.exists()


def test_t2_only_selected_tracked_bytes_reach_the_export(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    payload = build_code_export(_config(repo, out))
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and ".git" not in p.relative_to(out).parts)
    assert files == EXPECTED
    for absent in ("ignored.txt", ".env", "src/pkg/untracked.py", "src/private/b.py", "notes/plan.md"):
        assert not (out / absent).exists(), absent
    assert payload["file_count"] == len(EXPECTED)
    assert payload["secret_values_recorded"] is False
    assert _git(out, "rev-list", "--count", "HEAD").strip() == "1"


def test_t3_the_manifest_is_read_at_the_source_commit(repo: Path, tmp_path: Path) -> None:
    before = build_code_export(_config(repo, tmp_path / "one"))
    (repo / "code_export.toml").write_text(MANIFEST.replace('"docs/", ', ""), encoding="utf-8")
    after = build_code_export(_config(repo, tmp_path / "two"))
    assert after["tree_sha256"] == before["tree_sha256"]
    assert after["file_count"] == before["file_count"]


def test_t7_same_commit_and_identity_give_the_same_export_commit(repo: Path, tmp_path: Path) -> None:
    one = build_code_export(_config(repo, tmp_path / "one"))
    two = build_code_export(_config(repo, tmp_path / "two"))
    assert one["export_commit"] == two["export_commit"]
    assert _git(tmp_path / "one", "remote").strip() == ""
    log = _git(tmp_path / "one", "log", "-1", "--format=%an <%ae>|%cn <%ce>")
    assert log.strip() == f"{NAME} <{EMAIL}>|{NAME} <{EMAIL}>"


@pytest.mark.parametrize(
    ("case", "reason"),
    [("out_exists", "out_exists"), ("unknown_commit", "unknown_commit"), ("marker_email", "identity_refused"),
     ("no_manifest", "manifest_missing"), ("include_missing", "include_missing")],
)
def test_t7_refusals(repo: Path, tmp_path: Path, case: str, reason: str) -> None:
    out = tmp_path / "out"
    config = _config(repo, out)
    if case == "out_exists":
        out.mkdir()
    elif case == "unknown_commit":
        config = CodeExportConfig(repo, "0" * 40, out, NAME, EMAIL)
    elif case == "marker_email":
        config = CodeExportConfig(repo, "HEAD", out, NAME, "someone@" + "gmail.com")  # built at run time
    elif case == "no_manifest":
        _git(repo, "rm", "-q", "code_export.toml")
        _git(repo, "commit", "-q", "-m", "drop manifest")
    else:
        (repo / "code_export.toml").write_text(MANIFEST.replace('"README.md"', '"LICENSE"'), encoding="utf-8")
        _git(repo, "commit", "-q", "-am", "name a missing file")
    with pytest.raises(CodeExportError) as excinfo:
        build_code_export(config)
    assert excinfo.value.reason == reason
    if case != "out_exists":
        assert not out.exists()


def test_t7_cli_refusal_exits_2(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    result = _run("--project-root", str(repo), "--out", str(out), "--author-name", NAME, "--author-email", EMAIL)
    assert result.returncode == 2
    assert "out_exists" in result.stdout


def test_selection_excludes_and_sorts(repo: Path) -> None:
    assert selected_files(repo, "HEAD") == EXPECTED


@needs_gitleaks
def test_t9_a_clean_export_passes_every_gate(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build_code_export(_config(repo, out))
    assert check_code_export(out, env={}) == []
    result = _run("--check-only", str(out))
    assert result.returncode == 0, result.stdout
    receipt = tmp_path / "r.json"
    result = _run("--project-root", str(repo), "--out", str(tmp_path / "again"), "--author-name", NAME,
                  "--author-email", EMAIL, "--receipt-path", str(receipt))
    assert result.returncode == 0, result.stdout
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["gates"] == {"docs_links": 0, "private_markers": 0, "secrets": 0}


@needs_gitleaks
@pytest.mark.parametrize("plant", ["home_path", "fake_key", "broken_link"])
def test_t9_each_gate_catches_its_planted_case(repo: Path, tmp_path: Path, plant: str) -> None:
    if plant == "home_path":
        (repo / "src" / "pkg" / "a.py").write_text("ROOT = '/" + "Users" + "/alice/work'\n", encoding="utf-8")
        gate = "private_markers"
    elif plant == "fake_key":
        # A key-shaped value built at run time, so this test file never holds one.
        (repo / "src" / "pkg" / "a.py").write_text("KEY = '" + "AKIA" + "Q3EGRZ7XKMB4WN2P" + "'\n", encoding="utf-8")
        gate = "secrets"
    else:
        (repo / "docs" / "index.md").write_text("[gone](../notes/plan.md)\n", encoding="utf-8")
        gate = "docs_links"
    _git(repo, "commit", "-q", "-am", f"plant {plant}")
    out = tmp_path / "out"
    build_code_export(_config(repo, out))
    findings = check_code_export(out, env={})
    assert {item["gate"] for item in findings} == {gate}, findings
    assert _run("--check-only", str(out)).returncode == 1


@needs_gitleaks
def test_t9_a_gitleaks_run_that_fails_is_never_clean(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build_code_export(_config(repo, out))
    (out / ".gitleaks.toml").write_text("this is [not toml\n", encoding="utf-8")
    findings = check_code_export(out, env={})
    assert any(item["gate"] == "secrets" and "gitleaks" in item["detail"] for item in findings), findings


# --- review round phase-7-r1 (Codex): one regression test per finding ---

def test_r1_f1_a_force_tracked_ignored_file_is_in_the_export_commit(repo: Path, tmp_path: Path) -> None:
    # The export's own .gitignore must not drop a selected, tracked file from the commit.
    (repo / ".gitignore").write_text("ignored.txt\n.env\n*.log\n", encoding="utf-8")
    (repo / "src" / "pkg" / "kept.log").write_text("selected and tracked\n", encoding="utf-8")
    _git(repo, "add", "-f", "src/pkg/kept.log")
    _git(repo, "commit", "-q", "-am", "force-track a file the export's .gitignore matches")
    out = tmp_path / "out"
    payload = build_code_export(_config(repo, out))
    committed = sorted(_git(out, "ls-files").split())
    assert "src/pkg/kept.log" in committed
    assert payload["file_count"] == len(committed)


def test_r1_f2_a_receipt_inside_the_export_is_refused(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = _run("--project-root", str(repo), "--out", str(out), "--author-name", NAME, "--author-email", EMAIL,
                  "--receipt-path", str(out / "README.md"))
    assert result.returncode == 2, result.stdout
    assert "receipt_inside_export" in result.stdout
    assert not out.exists()


@needs_gitleaks
def test_r1_f3_a_relative_receipt_path_is_resolved(repo: Path, tmp_path: Path) -> None:
    work = tmp_path / "cwd"
    work.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(repo), "--out", str(tmp_path / "out"),
         "--author-name", NAME, "--author-email", EMAIL, "--receipt-path", "receipt.json"],
        cwd=work, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((work / "receipt.json").read_text(encoding="utf-8"))["file_count"] == len(EXPECTED)


def test_r1_f4_a_commit_that_is_not_an_ancestor_of_head_is_refused(repo: Path, tmp_path: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "src" / "pkg" / "side.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "src/pkg/side.py")
    _git(repo, "commit", "-q", "-m", "a sibling-branch commit")
    side = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "main")
    with pytest.raises(CodeExportError) as excinfo:
        build_code_export(CodeExportConfig(repo, side, tmp_path / "out", NAME, EMAIL))
    assert excinfo.value.reason == "not_an_ancestor"
    assert not (tmp_path / "out").exists()


def test_r1_f1_a_commit_that_would_differ_from_the_selection_is_refused(repo: Path, tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import code_export

    real = code_export._git

    def drop_one(root, *args, **kwargs):
        out = real(root, *args, **kwargs)
        if args[:1] == ("ls-files",) and root != repo:
            return b"\0".join(out.split(b"\0")[1:])  # the staged index lost a selected file
        return out

    monkeypatch.setattr(code_export, "_git", drop_one)
    out = tmp_path / "out"
    with pytest.raises(CodeExportError) as excinfo:
        build_code_export(_config(repo, out))
    assert excinfo.value.reason == "commit_mismatch"
    assert not out.exists()
    assert not any(p.name.startswith(".code-export-") for p in tmp_path.iterdir())


# --- review round pr-43-r1 (Codex): one regression test per finding ---

# A home path built at run time, so no line of this file reads as one.
HOME_PATH = "/" + "Users" + "/alice/private-note"


@pytest.mark.parametrize(
    ("name", "data", "flagged"),
    [
        ("nul-led.bin", b"\0" + HOME_PATH.encode() + b"\n", True),
        ("big.txt", HOME_PATH.encode() + b"\n" + b"x" * (3 * 1024 * 1024), True),
        ("clean.bin", b"\0\x01\x02 no marker here", False),
        ("clean-big.txt", b"x" * (3 * 1024 * 1024), False),
    ],
    ids=["nul-led", "oversized", "clean-binary", "clean-oversized"],
)
def test_pr43_f1_files_the_text_scan_skips_meet_the_byte_rules(
    repo: Path, tmp_path: Path, monkeypatch, name: str, data: bytes, flagged: bool
) -> None:
    from conference_pipeline.publishing import code_export

    (repo / "src" / "pkg" / name).write_bytes(data)
    _git(repo, "add", f"src/pkg/{name}")
    _git(repo, "commit", "-q", "-m", f"plant {name}")
    out = tmp_path / "out"
    build_code_export(_config(repo, out))
    monkeypatch.setattr(code_export, "_gitleaks_history", lambda _out: [])
    findings = check_code_export(out, env={})
    marked = [item for item in findings if item["gate"] == "private_markers"]
    if flagged:
        assert [item["path"] for item in marked] == [f"src/pkg/{name}"], findings
    else:
        assert findings == [], findings


def test_pr43_f1_skipped_symlinks_and_unreadable_entries_never_pass(tmp_path: Path) -> None:
    # git archive's tar filter already refuses a symlink that leaves the export, so the
    # build never carries one; the gate still reads every skip it is handed.
    from conference_pipeline.publishing.code_export import _skipped_file_findings

    (tmp_path / "home").symlink_to(HOME_PATH)
    (tmp_path / "inside").symlink_to("README.md")
    skipped = [("home", "not_a_file"), ("inside", "not_a_file"), ("gone", "not_a_file"), (".git/x", "binary")]
    found = _skipped_file_findings(tmp_path, skipped, rules=None)
    assert [(item["path"], item["detail"]) for item in found] == [
        ("home", "symlink target holds a private marker"),
        ("gone", "unscannable (not_a_file)"),
    ]


def test_pr43_the_email_rule_is_linear_on_a_long_line() -> None:
    import time

    from conference_pipeline._security.private_markers import builtin_rules, has_publishing_marker

    line = "x" * 400_000
    started = time.monotonic()
    assert not has_publishing_marker(line, rules=builtin_rules())
    assert has_publishing_marker(line + " " + "someone" + "@" + "corp.io", rules=builtin_rules())
    assert time.monotonic() - started < 5  # it took minutes before the rule was anchored


def test_pr43_f2_a_receipt_reaching_the_export_through_dotdot_is_refused(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    (tmp_path / "sibling").mkdir()
    receipt = tmp_path / "sibling" / ".." / "out" / "README.md"
    result = _run("--project-root", str(repo), "--out", str(out), "--author-name", NAME, "--author-email", EMAIL,
                  "--receipt-path", str(receipt))
    assert result.returncode == 2, result.stdout
    assert "receipt_inside_export" in result.stdout
    assert not out.exists()


@pytest.mark.parametrize("report", ["[null]", "[17]", '[{"RuleID": "x"}, "text"]'], ids=["null", "number", "mixed"])
def test_pr43_f3_a_gitleaks_report_with_a_non_object_entry_is_unreadable(tmp_path: Path, monkeypatch, report: str) -> None:
    from conference_pipeline.publishing import code_export

    def fake_run(args, **_kwargs):
        Path(args[args.index("--report-path") + 1]).write_text(report, encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(code_export.shutil, "which", lambda _name: "/usr/bin/gitleaks")
    monkeypatch.setattr(code_export.subprocess, "run", fake_run)
    assert code_export._gitleaks_history(tmp_path) == [{"gate": "secrets", "detail": "gitleaks_report_unreadable"}]


# --- review round phase-7-r2 (Codex): one regression test per finding ---

def test_r2_f1_the_email_rule_matches_exactly_what_the_unanchored_rule_matched() -> None:
    import random
    import re

    from conference_pipeline._security.private_markers import _EMAIL, builtin_rules, has_publishing_marker, redact_text

    rules = builtin_rules()
    email = next(rule.pattern for rule in rules if rule.id == "email")
    plain = re.compile(_EMAIL)
    second = "+secret" + "@" + "corp.io"
    joined = "a" + "@" + "example.com" + second  # a fixture address, then a real one with no gap
    assert [m.span() for m in email.finditer(joined)] == [m.span() for m in plain.finditer(joined)]
    assert has_publishing_marker(joined, rules=rules)
    assert second not in redact_text(joined, rules)
    rng = random.Random(43)
    alphabet = "ab1.-_+%@ Z/:x"
    for _ in range(20_000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert [m.span() for m in email.finditer(text)] == [m.span() for m in plain.finditer(text)], text


def test_r2_f2_a_receipt_reaching_the_export_through_a_symlink_and_dotdot_is_refused(repo: Path, tmp_path: Path) -> None:
    work, elsewhere = tmp_path / "work", tmp_path / "elsewhere" / "sub"
    work.mkdir()
    elsewhere.mkdir(parents=True)
    (work / "link").symlink_to(elsewhere)
    out = work / "export"
    receipt = work / "link" / ".." / "export" / "README.md"  # the writer collapses ".." first
    result = _run("--project-root", str(repo), "--out", str(out), "--author-name", NAME, "--author-email", EMAIL,
                  "--receipt-path", str(receipt))
    assert result.returncode == 2, result.stdout
    assert "receipt_inside_export" in result.stdout
    assert not out.exists()


def test_r2_f3_dangling_symlinks_and_fifos_in_the_tree_never_pass(repo: Path, tmp_path: Path, monkeypatch) -> None:
    import os

    from conference_pipeline.publishing import code_export

    out = tmp_path / "out"
    build_code_export(_config(repo, out))
    (out / "src" / "pkg" / "dangling").symlink_to(HOME_PATH)
    os.mkfifo(out / "src" / "pkg" / "pipe")
    monkeypatch.setattr(code_export, "_gitleaks_history", lambda _out: [])
    findings = check_code_export(out, env={})
    assert sorted((item["path"], item["detail"]) for item in findings) == [
        ("src/pkg/dangling", "symlink target holds a private marker"),
        ("src/pkg/pipe", "unscannable (not_a_file)"),
    ], findings


# --- review round phase-7-r3 (Codex): one regression test per finding ---

def _address() -> str:
    return "secret" + "@" + "corp.io"


def test_r3_f1_a_symlinked_scan_root_is_scanned_through(repo: Path, tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline._security.private_markers import builtin_rules, load_allowlist, scan_paths
    from conference_pipeline.publishing import code_export

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "report.txt").write_text(_address() + "\n", encoding="utf-8")
    (tmp_path / "alias").symlink_to(tree)
    report = scan_paths(tmp_path / "alias", ["."], builtin_rules(), load_allowlist(None), tracked_only=False)
    assert [(f.path, f.rule) for f in report.findings] == [("report.txt", "email")]

    out = tmp_path / "out"
    build_code_export(_config(repo, out))
    (out / "src" / "pkg" / "note.txt").write_text(HOME_PATH + "\n", encoding="utf-8")
    (tmp_path / "out-alias").symlink_to(out)
    monkeypatch.setattr(code_export, "_gitleaks_history", lambda _out: [])
    findings = check_code_export(tmp_path / "out-alias", env={})
    assert [(item["gate"], item["path"]) for item in findings] == [("private_markers", "src/pkg/note.txt")], findings


def test_r3_f2_a_root_through_a_symlink_and_dotdot_scans_where_reads_land(tmp_path: Path) -> None:
    from conference_pipeline._security.private_markers import builtin_rules, load_allowlist, scan_paths

    (tmp_path / "data" / "sub").mkdir(parents=True)
    (tmp_path / "data" / "bundle").mkdir()
    (tmp_path / "data" / "bundle" / "report.txt").write_text(_address() + "\n", encoding="utf-8")
    (tmp_path / "work" / "bundle").mkdir(parents=True)  # the lexical reading of the path: empty
    (tmp_path / "work" / "link").symlink_to(tmp_path / "data" / "sub")
    root = tmp_path / "work" / "link" / ".." / "bundle"  # the filesystem reading: data/bundle
    report = scan_paths(root, ["."], builtin_rules(), load_allowlist(None), tracked_only=False)
    assert [(f.path, f.rule) for f in report.findings] == [("report.txt", "email")]


# --- review round phase-7-r4 (Codex): one regression test per finding ---

def test_r4_f1_a_receipt_differing_from_the_export_only_in_case_is_refused(repo: Path, tmp_path: Path) -> None:
    # On a case-insensitive volume (the macOS default) these name the same directory.
    out = tmp_path / "Export"
    result = _run("--project-root", str(repo), "--out", str(out), "--author-name", NAME, "--author-email", EMAIL,
                  "--receipt-path", str(tmp_path / "export" / "receipt.json"))
    assert result.returncode == 2, result.stdout
    assert "receipt_inside_export" in result.stdout
    assert not out.exists()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads unreadable directories")
def test_r4_f2_an_unreadable_directory_is_a_finding(repo: Path, tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import code_export

    out = tmp_path / "out"
    build_code_export(_config(repo, out))
    locked = out / "src" / "locked"
    locked.mkdir()
    (locked / "note.txt").write_text(HOME_PATH + "\n", encoding="utf-8")
    locked.chmod(0)
    try:
        monkeypatch.setattr(code_export, "_gitleaks_history", lambda _out: [])
        findings = check_code_export(out, env={})
    finally:
        locked.chmod(0o755)
    assert [(item["gate"], item["path"]) for item in findings] == [("private_markers", "src/locked")], findings


def test_r4_f1_the_post_build_check_compares_directories_by_identity(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_code_repo_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out"
    (out / "deep").mkdir(parents=True)
    (tmp_path / "alias").symlink_to(out)
    assert module._same_directory_above(tmp_path / "alias" / "deep" / "r.json", out)
    assert module._same_directory_above(out / "r.json", out)
    assert not module._same_directory_above(tmp_path / "elsewhere" / "r.json", out)
