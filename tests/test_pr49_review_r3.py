"""v6.0 Phase 7.1, PR 49 review pr-49-r3: one regression test per finding (F1-F2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_pr49_review_r1 import _hub_project

MARKER = "codename-zephyr-fixture"
ENV = {"CONFERENCE_PIPELINE_PRIVATE_MARKERS": MARKER}


def _binary(size: int) -> bytes:
    body = b"\x00\xff" + MARKER.encode() + b"\x81"
    return body + b"\x00" * max(0, size - len(body))


# --- F1: the env file's literal markers reach binary bytes too --------------------------


def test_f1_extra_markers_are_found_in_binary_bytes() -> None:
    from conference_pipeline._security.private_markers import extra_rules_from_env, has_publishing_marker_bytes

    rules = extra_rules_from_env(ENV)
    assert has_publishing_marker_bytes(_binary(64), rules=rules) is True
    # Control: binary bytes without the literal stay clean.
    assert has_publishing_marker_bytes(b"\x00\xffclean\x81", rules=rules) is False


@pytest.mark.parametrize("size", [64, 2 * 1024 * 1024 + 1])
def test_f1_hub_refuses_a_binary_asset_holding_an_env_marker(tmp_path: Path, monkeypatch, size: int) -> None:
    from conference_pipeline.publishing.release_hub import hub_findings
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "asset.bin").write_bytes(_binary(size))
    found = hub_findings(hub, secret_scanner=lambda paths: [], environ=ENV)
    assert any(f.startswith("private_marker: asset.bin") for f in found), found


# --- F2: hub check and hub build load the env file publish loads ------------------------


def _env_file(tmp_path: Path) -> Path:
    env_file = tmp_path / "hub.env"
    env_file.write_text(f"CONFERENCE_PIPELINE_PRIVATE_MARKERS={MARKER}\n", encoding="utf-8")
    return env_file


def test_f2_hub_check_applies_markers_from_its_env_file(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import release_hub
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "notes.txt").write_text(MARKER + "\n", encoding="utf-8")
    lines: list[str] = []
    code = release_hub.hub_check(
        hub, project_root=_hub_project(tmp_path), env_file=_env_file(tmp_path), environ={},
        secret_scanner=lambda paths: [], echo=lines.append,
    )
    assert code == 1 and any("private_marker: notes.txt" in line for line in lines), lines


def test_f2_hub_check_without_the_marker_still_exits_0(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing import release_hub
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    code = release_hub.hub_check(
        hub, project_root=_hub_project(tmp_path), env_file=_env_file(tmp_path), environ={},
        secret_scanner=lambda paths: [], echo=lambda s: None,
    )
    assert code == 0


def test_f2_the_check_command_takes_the_env_file_and_root() -> None:
    import inspect

    from conference_pipeline.publishing.release_hub import build_command, check_command

    for command in (check_command, build_command):
        names = set(inspect.signature(command).parameters)
        assert {"project_root", "env_file"} <= names, (command.__name__, names)


# --- release check meets the same binary gap (recorded at pr-49-r2, closed here) --------

from tests.test_release_cli import SLUG, _no_secrets, cli, w0, w4  # noqa: E402,F401


@pytest.mark.parametrize("with_marker", [True, False])
def test_release_check_applies_env_markers_to_binary_files(w4, with_marker: bool) -> None:
    (w4.bundle / "asset.bin").write_bytes(_binary(64))
    env = ENV if with_marker else {}
    cli().release_check(SLUG, env=env, secret_scanner=_no_secrets, **w4.kw())
    flagged = "asset.bin: publishing marker in a binary file" in w4.text() or any(
        "asset.bin" in line and "marker" in line for line in w4.out
    )
    assert flagged is with_marker, w4.text()


# --- the pr-49-r2 F1 hub repair read a clean PDF with the literal ".env" token ----------


def test_the_hub_byte_check_does_not_flag_a_clean_pdf(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing.release_hub import _marker_findings
    from tests.test_phase71_review_r3 import _hub
    from tests.test_release_reader_families import _ascii_pdf

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "paper.pdf").write_bytes(_ascii_pdf("Adaptive pipelines and changes.environmental changes"))
    assert not [f for f in _marker_findings(hub, ENV) if "paper.pdf" in f]
    (hub / "paper.pdf").write_bytes(_ascii_pdf("Adaptive pipelines and " + MARKER))
    assert [f for f in _marker_findings(hub, ENV) if "paper.pdf" in f]
