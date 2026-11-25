"""The retired ICML 2026 post-14A supervisor refuses and points at the documented chain."""

from __future__ import annotations

import subprocess

import pytest

from scripts import supervise_icml2026_post_14a as supervisor


def test_retired_supervisor_refuses_without_running_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError(f"retired supervisor ran a subprocess: {args}")

    monkeypatch.setattr(subprocess, "run", must_not_run)
    monkeypatch.setattr(subprocess, "Popen", must_not_run)

    assert supervisor.main(["--workspace", "conferences/icml2026"]) == 2
    err = capsys.readouterr().err
    assert "retired" in err
    assert "docs/workflows/pipeline-workflow.md" in err


def test_retired_supervisor_prints_the_post_full_read_order() -> None:
    order = supervisor.POST_FULL_READ_ORDER
    labels = ["07A", "07B", "07C", "paper matching", "pass 1", "07D-07H", "pass 2", "Stage 8"]
    positions = [next(i for i, step in enumerate(order) if label in step) for label in labels]
    assert positions == sorted(positions)
    assert "WITHOUT --provider-cost-approved" in order[positions[4]]
    assert "WITH --provider-cost-approved" in order[positions[6]]
    assert "research_engine" in order[positions[2]]


def test_documented_chain_section_exists() -> None:
    from pathlib import Path

    doc = (Path(__file__).resolve().parent.parent / "docs" / "workflows" / "pipeline-workflow.md").read_text(
        encoding="utf-8"
    )
    assert "### Post-Full-Read Analysis Chain" in doc
