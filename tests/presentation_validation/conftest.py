"""Pytest fixtures for presentation validation contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.presentation_validation.fixtures import make_validation_workspace


@pytest.fixture
def validation_workspace(tmp_path: Path):
    return make_validation_workspace(tmp_path)
