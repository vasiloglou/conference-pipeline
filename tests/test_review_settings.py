"""Per-conference review settings: who reviews a finished deck, beyond the default panel."""

from __future__ import annotations

from pathlib import Path

import pytest

from conference_pipeline.model_governance import ModelGovernanceError
from conference_pipeline.presentation import presentation_review as pr
from conference_pipeline.presentation.review_settings import (
    ReviewSettingsError,
    advisory_visual_qc_selection,
    configured_reviewer_tokens,
    load_review_settings,
)

K3 = "openrouter/moonshotai/kimi-k3"


def _workspace(tmp_path: Path, body: str) -> Path:
    workspace = tmp_path / "conferences" / "demo2026"
    workspace.mkdir(parents=True)
    (workspace / "config.yaml").write_text(body, encoding="utf-8")
    return workspace


def test_without_a_section_nothing_changes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PRESENTATION_REVIEW_REVIEWERS", raising=False)
    workspace = _workspace(tmp_path, "conference:\n  slug: demo\n")
    assert load_review_settings(workspace) == {}
    assert configured_reviewer_tokens(workspace) is None
    assert advisory_visual_qc_selection(workspace) is None
    assert pr.configured_reviewers(workspace=workspace) == pr.DEFAULT_REVIEWERS
    assert pr.configured_reviewers(workspace=None) == pr.DEFAULT_REVIEWERS
    assert load_review_settings(tmp_path / "does-not-exist") == {}


def test_the_conference_config_names_the_panel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PRESENTATION_REVIEW_REVIEWERS", raising=False)
    workspace = _workspace(
        tmp_path,
        f"presentation_review:\n  reviewers:\n    - claude_code_opus\n    - codex_5_5\n    - gemini\n    - {K3}\n",
    )
    panel = pr.configured_reviewers(workspace=workspace)
    assert [member.transport for member in panel] == ["cli", "cli", "cli", "api"]
    assert panel[-1].model == K3


def test_an_explicit_spec_and_the_environment_outrank_the_config(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path, f"presentation_review:\n  reviewers: [gemini, {K3}]\n")
    monkeypatch.setenv("PRESENTATION_REVIEW_REVIEWERS", "claude_code_opus,gemini")
    assert [m.reviewer_id for m in pr.configured_reviewers(workspace=workspace)] == ["claude_code_opus", "gemini"]
    assert [m.reviewer_id for m in pr.configured_reviewers("codex_5_5", workspace=workspace)] == ["codex_5_5"]
    monkeypatch.delenv("PRESENTATION_REVIEW_REVIEWERS")
    assert [m.reviewer_id for m in pr.configured_reviewers(workspace=workspace)][0] == "gemini"


def test_a_reviewer_the_registry_does_not_allow_fails_loudly_never_a_smaller_panel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PRESENTATION_REVIEW_REVIEWERS", raising=False)
    workspace = _workspace(tmp_path, "presentation_review:\n  reviewers: [gemini, openrouter/z-ai/glm-5.3]\n")
    with pytest.raises(ModelGovernanceError):
        pr.configured_reviewers(workspace=workspace)


def test_malformed_sections_are_errors(tmp_path: Path) -> None:
    with pytest.raises(ReviewSettingsError):
        configured_reviewer_tokens(_workspace(tmp_path, "presentation_review:\n  reviewers: gemini\n"))
    with pytest.raises(ReviewSettingsError):
        load_review_settings(_workspace(tmp_path / "b", "presentation_review: [a, b]\n"))
    with pytest.raises(ReviewSettingsError):
        advisory_visual_qc_selection(_workspace(tmp_path / "c", "presentation_review:\n  advisory_visual_qc: {}\n"))
    with pytest.raises(ReviewSettingsError):
        load_review_settings(_workspace(tmp_path / "d", "presentation_review: [unclosed\n"))


def test_the_advisory_pass_resolves_to_a_governed_selection(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, f"presentation_review:\n  advisory_visual_qc:\n    model: {K3}\n")
    assert advisory_visual_qc_selection(workspace) == ("openrouter", K3)


def test_an_advisory_model_the_registry_does_not_allow_is_refused(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "presentation_review:\n  advisory_visual_qc:\n    model: openrouter/z-ai/glm-5.3\n")
    with pytest.raises(ModelGovernanceError):
        advisory_visual_qc_selection(workspace)


@pytest.mark.repo_data("conferences/icml2026/config.yaml")
def test_the_icml_conference_config_is_valid_and_picks_up_kimi() -> None:
    root = Path(__file__).resolve().parents[1] / "conferences" / "icml2026"
    assert advisory_visual_qc_selection(root) == ("openrouter", K3)
    panel = pr.configured_reviewers(workspace=root)
    assert {m.transport for m in panel} == {"cli", "api"}


def test_a_conference_without_a_section_keeps_the_default_panel() -> None:
    root = Path(__file__).resolve().parents[1] / "conferences" / "cvpr2026"
    if (root / "config.yaml").is_file():
        assert load_review_settings(root) == {}
