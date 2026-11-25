"""Per-conference review settings: who reviews a finished deck, beyond the default panel.

Read from the ``presentation_review`` section of a conference's ``config.yaml``::

    presentation_review:
      reviewers: [claude_code_opus, codex_5_5, gemini, openrouter/moonshotai/kimi-k3]
      advisory_visual_qc:
        model: openrouter/moonshotai/kimi-k3

Absent, everything is as it was: the three CLI reviewers and no advisory pass.
Whether a model may take part is still the registry's decision -- this only
names the choice, and a name the registry does not allow is an error when it is
used, never a silently smaller panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SECTION = "presentation_review"


class ReviewSettingsError(ValueError):
    """The ``presentation_review`` section is present but malformed."""


def load_review_settings(workspace: Path | str | None) -> dict[str, Any]:
    """The conference's ``presentation_review`` section, or ``{}`` when there is none."""

    if workspace is None:
        return {}
    config_path = Path(workspace) / "config.yaml"
    if not config_path.is_file():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ReviewSettingsError(f"{config_path} is not valid YAML: {exc}") from exc
    section = payload.get(SECTION) if isinstance(payload, dict) else None
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ReviewSettingsError(f"{config_path}: {SECTION} must be a mapping")
    return section


def configured_reviewer_tokens(workspace: Path | str | None) -> str | None:
    """The configured panel as the comma-separated spec ``configured_reviewers`` takes, or None."""

    reviewers = load_review_settings(workspace).get("reviewers")
    if reviewers is None:
        return None
    if not isinstance(reviewers, list) or not all(isinstance(item, str) and item.strip() for item in reviewers):
        raise ReviewSettingsError(f"{SECTION}.reviewers must be a list of reviewer ids or model names")
    return ",".join(item.strip() for item in reviewers)


def advisory_visual_qc_selection(workspace: Path | str | None) -> tuple[str, str] | None:
    """(provider, model) of the advisory visual-QC pass, or None when none is configured.

    Validated against the visual-QC task's allowed models, so a typo or a model
    the registry has not allowed fails here, before any slide is sent anywhere.
    """

    section = load_review_settings(workspace).get("advisory_visual_qc")
    if section is None:
        return None
    if not isinstance(section, dict) or not (section.get("model") or section.get("provider")):
        raise ReviewSettingsError(f"{SECTION}.advisory_visual_qc needs a model (and optionally a provider)")
    from conference_pipeline.model_governance import validate_selection
    from conference_pipeline.model_governance.llm_client import resolve_selection

    provider, model = resolve_selection("presentation.visual_qc", section.get("provider"), section.get("model"))
    validate_selection("presentation.visual_qc", provider=provider, model=model)
    return provider, model
