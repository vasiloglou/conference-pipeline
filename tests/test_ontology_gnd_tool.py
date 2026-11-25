"""Phase 1 (v6.1-ICML 2026 Ontology): the total GND wrapper, against fake launchers."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.gnd_tool import (
    NOT_STARTED,
    TIMED_OUT,
    CommandFamily,
    GndUnavailable,
)


def _launcher(bin_dir: Path, body: str, *, shebang: str = "#!/bin/sh") -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / "gnd"
    path.write_text(f"{shebang}\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _env(bin_dir: Path, **extra: str) -> dict[str, str]:
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", **extra}


# T1
def test_missing_binary_is_reported_not_raised(tmp_path):
    env = {"PATH": str(tmp_path / "empty")}
    with pytest.raises(GndUnavailable):
        gnd_tool.gnd_executable(env=env)
    result = gnd_tool.run_gnd(["check"], family=CommandFamily.CHECK, cwd=tmp_path, env=env)
    assert result.exit_code == NOT_STARTED
    assert result.meaning == "not-started"


# T2
def test_require_gnd_names_both_versions_on_mismatch(tmp_path):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, 'echo "gnd 0.3.0"')
    with pytest.raises(GndUnavailable) as excinfo:
        gnd_tool.require_gnd("0.4.0", env=_env(bin_dir))
    assert "0.3.0" in str(excinfo.value) and "0.4.0" in str(excinfo.value)


# T3
@pytest.mark.parametrize(
    "code,meaning",
    [(0, "pass"), (1, "alethic-failure"), (2, "vacuous"), (3, "input-refused"), (4, "unchecked"), (7, "unmapped")],
)
def test_check_family_meanings_keep_the_raw_exit(tmp_path, code, meaning):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, f"exit {code}")
    result = gnd_tool.run_gnd(["check"], family=CommandFamily.CHECK, cwd=tmp_path, env=_env(bin_dir))
    assert result.exit_code == code
    assert result.meaning == meaning


# T3b
@pytest.mark.parametrize(
    "family,code,meaning",
    [
        (CommandFamily.VERDICT, 2, "support-rejected"),
        (CommandFamily.CHECK, 2, "vacuous"),
        (CommandFamily.WITNESS, 2, "missing-unknown-or-unsupported"),
        (CommandFamily.WHY_NOT, 2, "unsupported-rule"),
        (CommandFamily.ASSURANCE, 1, "stage-failed"),
        (CommandFamily.ASSURANCE, 2, "unmapped"),
    ],
)
def test_the_same_exit_means_different_things_per_family(family, code, meaning):
    assert gnd_tool.exit_meaning(family, code) == meaning


# T3c
@pytest.mark.parametrize(
    "code,meaning",
    [(0, "proved"), (1, "proof-invalid"), (2, "missing-unknown-or-unsupported"), (3, "input-invalid"), (4, "work-limit")],
)
def test_witness_family_meanings(code, meaning):
    assert gnd_tool.exit_meaning(CommandFamily.WITNESS, code) == meaning


# T4
def test_timeout_is_reported_and_the_process_killed(tmp_path):
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "finished"
    _launcher(bin_dir, f"sleep 5; touch {marker}")
    result = gnd_tool.run_gnd(["check"], family=CommandFamily.CHECK, cwd=tmp_path, timeout_s=0.5, env=_env(bin_dir))
    assert result.exit_code == TIMED_OUT
    assert result.meaning == "timeout"
    assert not marker.exists()


# T5
def test_undecodable_and_non_json_output(tmp_path):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, r"printf 'not json \377\376'")
    result = gnd_tool.run_gnd(["check"], family=CommandFamily.CHECK, cwd=tmp_path, env=_env(bin_dir))
    assert result.exit_code == 0
    assert "�" in result.stdout
    assert gnd_tool.parse_json_stdout(result) is None


# T6
@pytest.mark.parametrize("module", ["os", "literature;rm", "literature.Reading", "../literature"])
def test_example_module_is_validated_before_any_process(tmp_path, module):
    with pytest.raises(ValueError):
        gnd_tool.run_gnd_example(module, [], family=CommandFamily.OTHER, cwd=tmp_path,
                                 env={"PATH": str(tmp_path / "empty")})


def test_example_module_runs_with_the_launcher_interpreter(tmp_path):
    bin_dir = tmp_path / "bin"
    fake_python = tmp_path / "toolpy" / "python"
    fake_python.parent.mkdir()
    fake_python.write_text('#!/bin/sh\necho "$@"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    _launcher(bin_dir, "exit 0", shebang=f"#!{fake_python}")
    result = gnd_tool.run_gnd_example("literature.reading", ["replay", "r"], family=CommandFamily.REPLAY,
                                      cwd=tmp_path, env=_env(bin_dir))
    assert result.exit_code == 0 and result.meaning == "replayed"
    assert result.stdout.split() == ["-I", "-m", "gnd.examples.literature.reading", "replay", "r"]


# T6b
def test_launcher_without_shebang_is_unavailable(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gnd").write_text("print('no shebang')\n", encoding="utf-8")
    (bin_dir / "gnd").chmod(0o755)
    with pytest.raises(GndUnavailable):
        gnd_tool.gnd_python(env=_env(bin_dir))


# T6c
@pytest.mark.parametrize(
    "body,timeout,cause",
    [
        (None, 60, "did not start"),
        ("sleep 5", 0.5, "timed out"),
        ('echo "gnd banana"', 60, "unrecognised"),
        (r"printf 'gnd \377\n'", 60, "unrecognised"),
        ('echo "gnd 0.4.0"; exit 1', 60, "exited 1"),
    ],
)
def test_gnd_version_is_total(tmp_path, body, timeout, cause):
    bin_dir = tmp_path / "bin"
    if body is None:
        env = {"PATH": str(tmp_path / "empty")}
    else:
        _launcher(bin_dir, body)
        env = _env(bin_dir)
    with pytest.raises(GndUnavailable) as excinfo:
        gnd_tool.gnd_version(env=env, timeout_s=timeout)
    assert cause in str(excinfo.value)


def test_gnd_version_reads_the_version(tmp_path):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, 'echo "gnd 0.4.0"')
    assert gnd_tool.gnd_version(env=_env(bin_dir)) == "0.4.0"


# T6d
def test_require_gnd_without_gnd_is_unavailable_not_a_mismatch(tmp_path):
    with pytest.raises(GndUnavailable) as excinfo:
        gnd_tool.require_gnd("0.4.0", env={"PATH": str(tmp_path / "empty")})
    assert "requires" not in str(excinfo.value)


# T6e
def test_parse_json_stdout_object_and_non_object():
    def result(stdout: str):
        return gnd_tool.GndResult(("gnd",), CommandFamily.OTHER, 0, "success", stdout, "", 0.0)

    assert gnd_tool.parse_json_stdout(result('{"a": 1}')) == {"a": 1}
    assert gnd_tool.parse_json_stdout(result("[1, 2]")) is None


# T6f
def test_require_gnd_returns_the_version_on_match(tmp_path):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, 'echo "gnd 0.4.0"')
    assert gnd_tool.require_gnd("0.4.0", env=_env(bin_dir)) == "0.4.0"


# T9 (wrapper half): credentials never reach the subprocess
def test_credentials_are_scrubbed_from_the_subprocess_env(tmp_path):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, "env")
    env = _env(bin_dir, OPENAI_API_KEY="sk-x", HF_TOKEN="t", ICML_COOKIES_FILE="c", KEEP_ME="1")
    result = gnd_tool.run_gnd([], family=CommandFamily.OTHER, cwd=tmp_path, env=env)
    names = {line.split("=", 1)[0] for line in result.stdout.splitlines() if "=" in line}
    assert "PATH" in names and "KEEP_ME" not in names  # r1 F6: allowlist
    assert not names & {"OPENAI_API_KEY", "HF_TOKEN", "ICML_COOKIES_FILE"}


def test_scrubbed_env_defaults_to_the_process_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert "ANTHROPIC_API_KEY" not in gnd_tool.scrubbed_env()
    assert os.environ["ANTHROPIC_API_KEY"] == "x"


# --- review round 1 (notes/reviews/phase-1-r1) regressions ---------------------------

# r1 F6: an allowlist, not a suffix filter
def test_cloud_credentials_never_reach_the_subprocess(tmp_path):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, "env")
    env = _env(bin_dir, AWS_ACCESS_KEY_ID="a", AWS_SECRET_ACCESS_KEY="b", GOOGLE_APPLICATION_CREDENTIALS="c",
               AZURE_CLIENT_SECRET="d", PYTHONPATH="/x")
    result = gnd_tool.run_gnd([], family=CommandFamily.OTHER, cwd=tmp_path, env=env)
    names = {line.split("=", 1)[0] for line in result.stdout.splitlines() if "=" in line}
    assert not names & {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
                        "AZURE_CLIENT_SECRET"}
    assert {"PATH", "PYTHONPATH"} <= names


# r1 F7: a process killed by a signal is neither "not started" nor "timeout"
@pytest.mark.parametrize("signal_name,code", [("HUP", -1), ("INT", -2)])
def test_a_signal_exit_keeps_its_own_meaning(tmp_path, signal_name, code):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, f"kill -{signal_name} $$")
    result = gnd_tool.run_gnd([], family=CommandFamily.CHECK, cwd=tmp_path, env=_env(bin_dir))
    assert result.exit_code == code
    assert result.meaning == f"signal-{-code}"


# r1 F8: any launch error is a non-start, and gnd_version stays total
def test_a_launch_error_is_a_non_start(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    _launcher(bin_dir, 'echo "gnd 0.4.0"')

    def refuse(*args, **kwargs):
        raise PermissionError("not allowed")

    monkeypatch.setattr(gnd_tool.subprocess, "run", refuse)
    result = gnd_tool.run_gnd(["check"], family=CommandFamily.CHECK, cwd=tmp_path, env=_env(bin_dir))
    assert result.exit_code == NOT_STARTED and "not allowed" in result.stderr
    with pytest.raises(GndUnavailable):
        gnd_tool.gnd_version(env=_env(bin_dir))
