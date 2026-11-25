"""Corpus for the private-marker scanner (v6.0-Public-Release Phase 2, spec §8 T1).

Cases C1-C12 are the owner-approved approval instrument (spec D8); each test names its
case. The markers in tests/fixtures/private_markers/ are generic fixtures.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline._security.private_markers import (
    builtin_rules,
    extra_rules_from_env,
    load_allowlist,
    scan_paths,
    scan_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "tests" / "fixtures" / "private_markers"
CLI = REPO_ROOT / "scripts" / "audit_private_markers.py"
EMPTY = load_allowlist(None)


def _scan_fixture(name: str, rules=None):
    path = CORPUS / name
    return scan_text(path.read_text(encoding="utf-8"), name, rules or builtin_rules(), EMPTY)


def _rules(findings):
    return sorted(f.rule for f in findings)


def test_c01_users_home_path_is_found_and_redacted():
    findings = _scan_fixture("c01_home_users.py")
    assert _rules(findings) == ["home_path"]
    assert "<redacted:home_path>" in findings[0].excerpt
    assert "fixture-user" not in findings[0].excerpt
    assert findings[0].line == 1


def test_c02_linux_home_path_is_found():
    assert _rules(_scan_fixture("c02_home_linux.py")) == ["home_path"]


def test_c03_drive_account_and_cloud_storage_are_found():
    findings = _scan_fixture("c03_drive_account.py")
    assert _rules(findings) == ["cloud_storage", "drive_account"]
    assert all("someone@corp.test" not in f.excerpt for f in findings)


def test_c04_drive_glob_without_identity_is_clean():
    assert _scan_fixture("c04_drive_glob.py") == []


def test_c05_fixture_email_domains_are_clean():
    assert _scan_fixture("c05_fixture_emails.py") == []


def test_c06_real_domain_email_is_found():
    findings = _scan_fixture("c06_real_email.py")
    assert _rules(findings) == ["email"]
    assert "person@university.edu" not in findings[0].excerpt


def test_c07_shell_profile_read_counts_in_code_only():
    assert _rules(_scan_fixture("c07_shell_profile.py")) == ["shell_profile_secret"]
    assert _scan_fixture("c07_shell_profile.md") == []


def test_c08_drive_leaf_convention_is_clean():
    assert _scan_fixture("c08_drive_convention.md") == []


def _run_cli(*args, env_extra=None, cwd=REPO_ROOT):
    env = {k: v for k, v in os.environ.items() if k != "CONFERENCE_PIPELINE_PRIVATE_MARKERS"}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )


def test_c09_extra_literal_is_found_and_never_printed(tmp_path):
    secret = "SecretName"  # a stand-in for a maintainer literal
    (tmp_path / "a.py").write_text(f"OWNER = '{secret}'\n", encoding="utf-8")
    rules = extra_rules_from_env({"CONFERENCE_PIPELINE_PRIVATE_MARKERS": secret})
    findings = scan_text((tmp_path / "a.py").read_text(), "a.py", rules, EMPTY)
    assert _rules(findings) == ["extra"]
    assert secret not in findings[0].excerpt

    report_path = tmp_path / "report.json"
    result = _run_cli(
        "--tree", str(tmp_path), "--json", str(report_path),
        env_extra={"CONFERENCE_PIPELINE_PRIVATE_MARKERS": secret},
    )
    assert result.returncode == 1
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert secret not in report_path.read_text(encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["extra"] == 1


def test_c10_allowlist_suppresses_and_reports_stale_entries(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "c01.py").write_text('BASE = "/Users/fixture-user/repo"\n', encoding="utf-8")
    allow = tmp_path / "allow.toml"
    allow.write_text(
        '[[allow]]\npath = "c01.py"\nrule = "home_path"\nmatch = "/Users/fixture-user"\n'
        'reason = "fixture"\n\n'
        '[[allow]]\npath = "nothing/*.py"\nrule = "email"\nmatch = "x"\nreason = "unused"\n',
        encoding="utf-8",
    )
    report = scan_paths(
        tree, ["."], builtin_rules(), load_allowlist(allow), tracked_only=False
    )
    assert report.findings == []
    assert report.suppressed == 1
    assert [entry.path for entry in report.stale_allowlist] == ["nothing/*.py"]


def test_c11_cli_exit_codes(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "ok.py").write_text("print('hello')\n", encoding="utf-8")
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "bad.py").write_text('BASE = "/Users/fixture-user/repo"\n', encoding="utf-8")

    assert _run_cli("--tree", str(clean), "--no-env").returncode == 0
    result = _run_cli("--tree", str(dirty), "--no-env")
    assert result.returncode == 1
    assert "home_path" in result.stdout
    assert "fixture-user" not in result.stdout
    missing = _run_cli("--tree", str(clean), "--allowlist", str(tmp_path / "absent.toml"))
    assert missing.returncode == 2


def test_c12_binary_and_oversized_files_are_skipped(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01/Users/fixture-user\x00")
    (tmp_path / "big.txt").write_text("x" * (3 * 1024 * 1024) + "/Users/fixture-user\n")
    report = scan_paths(tmp_path, ["."], builtin_rules(), EMPTY, tracked_only=False)
    assert report.findings == []
    assert report.skipped == {"binary": 1, "oversized": 1}


@pytest.mark.parametrize("value", ["", " , ,"])
def test_extra_rules_ignore_empty_values(value):
    assert extra_rules_from_env({"CONFERENCE_PIPELINE_PRIVATE_MARKERS": value}) == ()


# ---------------------------------------------------------------------------
# Regression cases from review phase-2-r1 (successors of the approved C1-C12).
# ---------------------------------------------------------------------------


def test_c13_overlapping_matches_never_leak_an_extra_literal():  # review F1
    rules = builtin_rules() + extra_rules_from_env(
        {"CONFERENCE_PIPELINE_PRIVATE_MARKERS": "/Users/demo/SecretName,SecretName"}
    )
    findings = scan_text('P = "/Users/demo/SecretName"\n', "a.py", rules, EMPTY)
    assert findings
    assert all("SecretName" not in f.excerpt for f in findings)


def test_c14_paths_and_diagnostics_are_redacted(tmp_path):  # review F2
    secret = "SecretName"
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / f"{secret}.py").write_text(f"X = '{secret}'\n", encoding="utf-8")
    report_path = tmp_path / "r.json"
    env = {"CONFERENCE_PIPELINE_PRIVATE_MARKERS": secret}
    result = _run_cli("--tree", str(tree), "--json", str(report_path), env_extra=env)
    assert result.returncode == 1
    assert secret not in result.stdout + result.stderr
    assert secret not in report_path.read_text(encoding="utf-8")
    missing = _run_cli("--tree", str(tree), "--allowlist", str(tmp_path / f"{secret}.toml"), env_extra=env)
    assert missing.returncode == 2
    assert secret not in missing.stdout + missing.stderr


def test_c15_an_exemption_covers_only_the_match_it_names(tmp_path):  # review F3
    allow = tmp_path / "allow.toml"
    allow.write_text(
        '[[allow]]\npath = "t.py"\nrule = "home_path"\nmatch = "/Users/example/"\nreason = "fixture"\n',
        encoding="utf-8",
    )
    line = 'paths = ["/Users/example/a", "/Users/actual-owner/private"]\n'
    findings = scan_text(line, "t.py", builtin_rules(), load_allowlist(allow))
    assert _rules(findings) == ["home_path"]


def test_c16_drive_account_dedup_keeps_unrelated_emails():  # review F4
    line = "GoogleDrive-fixture@corp.test/x person@university.edu\n"
    assert _rules(scan_text(line, "a.py", builtin_rules(), EMPTY)) == ["drive_account", "email"]


@pytest.mark.parametrize("path", ["/home/_runner/private", "/Users/élise/private"])
def test_c17_home_names_need_not_start_alphanumeric(path):  # review F5
    assert _rules(scan_text(f'P = "{path}"\n', "a.py", builtin_rules(), EMPTY)) == ["home_path"]


@pytest.mark.parametrize("body", ["allow = [1]\n", 'allow = "x"\n', "[[allow]]\npath = 1\nrule = 2\nreason = 3\nmatch = []\n"])
def test_c18_malformed_allowlist_is_an_allowlist_error(tmp_path, body):  # review F6
    from conference_pipeline._security.private_markers import AllowlistError

    allow = tmp_path / "allow.toml"
    allow.write_text(body, encoding="utf-8")
    with pytest.raises(AllowlistError):
        load_allowlist(allow)
    clean = tmp_path / "clean"
    clean.mkdir()
    assert _run_cli("--tree", str(clean), "--allowlist", str(allow), "--no-env").returncode == 2


def test_c19_cli_exits_zero_when_the_allowlist_covers_every_finding(tmp_path):  # review C10 gap
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "c01.py").write_text('BASE = "/Users/fixture-user/repo"\n', encoding="utf-8")
    allow = tmp_path / "allow.toml"
    allow.write_text(
        '[[allow]]\npath = "c01.py"\nrule = "home_path"\nmatch = "/Users/fixture-user"\nreason = "fixture"\n',
        encoding="utf-8",
    )
    assert _run_cli("--tree", str(tree), "--allowlist", str(allow), "--no-env").returncode == 0


@pytest.mark.parametrize("text", ["reject `/Users/...` paths", "under /home/… or similar"])
def test_c20_dot_only_placeholders_are_not_names(text):  # follow-on of the F5 repair
    assert scan_text(text + "\n", "a.md", builtin_rules(), EMPTY) == []


# --- v6.0 Phase 3 (D4): the publishing gates share one marker definition ------------

LEGACY_TOKENS = ("file://", "/Users/", "/Volumes/", "Google Drive", "drive.google.com", ".env")
NEW_MARKERS = {
    "home_path": "see /home/fixture-user/notes.txt",
    "email": "contact someone@realdomain.org",
    "extra": "built by SecretOperatorName",
}


def _gate_blocks(gate: str, text: str, tmp_path: Path) -> bool:
    """True if the named publishing gate refuses or flags ``text``."""
    from conference_pipeline.publishing import (
        byok_tools,
        dataset_package,
        public_closeout,
        public_index,
        public_repo,
        self_host_kit,
        static_site,
    )

    root = tmp_path / gate
    root.mkdir(parents=True)
    path = root / "file.md"
    path.write_text(text + "\n", encoding="utf-8")
    try:
        if gate == "dataset_package":
            dataset_package.assert_no_private_markers(root)
        elif gate == "public_closeout":
            return public_closeout.scan_public_tree(root)["status"] != "clean"
        elif gate == "public_repo":
            return public_repo.text_has_private_marker(text)
        elif gate == "public_index":
            return public_index.text_has_private_marker(path)
        elif gate == "static_site":
            static_site.assert_site_output_public_safe(root)
        elif gate == "byok_tools":
            byok_tools.scan_browser_policy([path])
        elif gate == "self_host_kit":
            self_host_kit.assert_no_private_markers([path])
    except Exception:
        return True
    return False


GATES = ("dataset_package", "public_closeout", "public_repo", "public_index", "static_site", "byok_tools", "self_host_kit")
ENV_DOCUMENTING_GATES = {"byok_tools", "self_host_kit"}


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("token", LEGACY_TOKENS)
def test_t5_every_gate_still_blocks_each_legacy_token(gate, token, tmp_path, monkeypatch):
    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    # ".env" blocks as a file name: a letter or digit after it makes a word, not the file
    # (one rule for every scan, after pr-49-r6). The other tokens block as substrings.
    sample = f"prefix {token}/suffix" if token == ".env" else f"prefix {token}suffix"
    blocked = _gate_blocks(gate, sample, tmp_path)
    if token == ".env" and gate in ENV_DOCUMENTING_GATES:
        assert not blocked  # these two gates document a .env file on purpose
    else:
        assert blocked


@pytest.mark.parametrize("gate", GATES)
def test_t5_a_word_holding_env_passes_every_gate(gate, tmp_path, monkeypatch):
    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    assert not _gate_blocks(gate, "changes.environmental and the .envelope", tmp_path), gate


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("rule", sorted(NEW_MARKERS))
def test_t5_every_gate_now_blocks_the_shared_rules(gate, rule, tmp_path, monkeypatch):
    monkeypatch.setenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", "SecretOperatorName")
    assert _gate_blocks(gate, NEW_MARKERS[rule], tmp_path)


def test_t5_clean_text_passes_every_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    for gate in GATES:
        assert not _gate_blocks(gate, "plain public slide text, contact team@example.com", tmp_path / "c"), gate


def test_t5_hits_never_return_the_matched_text(monkeypatch):
    from conference_pipeline._security.private_markers import publishing_marker_hits, redact_publishing_markers

    monkeypatch.setenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", "SecretOperatorName")
    text = "SecretOperatorName at /home/fixture-user and file://x"
    hits = publishing_marker_hits(text)
    assert hits == ["file://", "extra", "home_path"]
    redacted = redact_publishing_markers(text)
    assert "SecretOperatorName" not in redacted and "fixture-user" not in redacted and "file://" not in redacted


def test_s2_an_exempted_env_mention_never_hides_a_home_path():  # IF8 follow-on
    from conference_pipeline._security.private_markers import publishing_marker_hits

    assert publishing_marker_hits("see /home/fixture-user/.env", allow_env_mention=True, rules=None) == ["home_path"]
    assert publishing_marker_hits("copy .env.example to .env", allow_env_mention=True) == []
    assert publishing_marker_hits("/Users/fixture-user/x") == ["/Users/"]


def test_has_publishing_marker_bytes_text_and_binary() -> None:
    """v6.0 Phase 4: text meets every rule whatever its name; binary meets the tokens."""
    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    email = ("someone" + "@" + "realmail.example-corp.com").encode()
    assert has_publishing_marker_bytes(b"contact " + email)  # UTF-8 text: the email rule
    assert not has_publishing_marker_bytes(b"\xff\xfe" + email)  # binary: tokens only
    assert has_publishing_marker_bytes(b"\xff\xfe https://drive." + b"google.com/x")
    assert has_publishing_marker_bytes(b"\xff\xfe conf/" + b".env ")
    assert not has_publishing_marker_bytes(b"\xff\xfe x" + b".envoy")


def test_c18_a_home_like_segment_inside_a_web_url_is_not_a_home_path():  # 6b, NeurIPS 2025 deck 19
    """A public page's URL path may hold /home/<x> (people.math.harvard.edu/~ctm/home/text/...);
    that is a URL, not a local home folder. A bare one, or one after a URL, still counts."""
    from conference_pipeline._security.private_markers import has_publishing_marker

    home = "/" + "home/" + "text"
    users = "/" + "Users/" + "alice"
    assert not has_publishing_marker(f'Target="https://people.example.edu/~ctm{home}/others/entropy.pdf"')
    assert has_publishing_marker(f"see {home}/notes.txt")
    assert has_publishing_marker(f"see https://a.example/ then {home}/notes.txt")
    assert has_publishing_marker(f"file://{users}/x")


def test_c19_markers_inside_a_zip_container_are_found():  # 6b, NeurIPS 2025 PowerPoint decks
    """A .pptx is a zip: compressed parts hide every marker from a byte scan."""
    import io
    import zipfile

    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    def pptx(rels: str) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("ppt/slides/_rels/slide1.xml.rels", rels)
            z.writestr("ppt/media/image1.png", b"\x89PNG" + bytes(range(256)) * 4)
        return buf.getvalue()

    drive = "https://drive." + "google.com/file/d/X/view"
    assert has_publishing_marker_bytes(pptx(f'<Relationship Target="{drive}"/>'))
    assert has_publishing_marker_bytes(pptx('<a>' + "/" + "Users/" + "alice/deck.md</a>"))
    assert not has_publishing_marker_bytes(pptx('<Relationship Target="https://people.example.edu/~ctm/home/text/x.pdf"/>'))
