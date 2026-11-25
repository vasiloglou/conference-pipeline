from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.test_exhaustive_topic_extraction import _build_workspace
from tests.presentation_validation.fixtures import make_validation_workspace


pytest_plugins = ("tests.live_llm._helpers",)

_LIVE_LLM_SKIP_REASON = (
    "requires RUN_LIVE_LLM=1 + a reachable local LLM endpoint "
    "(set EXTRACTION_LOCAL_LLM_BASE_URL for HTTP or install mlx-lm with a "
    "local model for MLX); see docs/reference/pipeline-capabilities.md"
)
_LOCAL_E2E_SKIP_REASON = (
    "v1.9 local lane end-to-end requires RUN_LOCAL_E2E=1 + Fake clients "
    "or real MLX/Ollama on Apple Silicon; see docs/local-llm.md"
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_data_missing(paths: tuple[str, ...], root: Path = _REPO_ROOT) -> str | None:
    """The first named repository path that is absent under ``root``, or None.

    ``@pytest.mark.repo_data(<paths>)`` marks a test that reads repository files the public
    code export does not carry (conference workspaces, ``.planning/`` history, brand assets).
    In the private checkout every path exists and the test runs as before; in the export it
    is skipped with the missing path named (v6.0 Phase 7, D2).
    """
    for rel in paths:
        if not (root / rel).exists():
            return rel
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        # A module whose test sources are fingerprinted names its paths in a REPO_DATA table
        # ({test function name: paths}) instead of a decorator; both feed the same rule.
        table = getattr(getattr(item, "module", None), "REPO_DATA", None) or {}
        tabled = table.get(getattr(item, "originalname", item.name))
        path_sets = [tuple(mark.args) for mark in item.iter_markers(name="repo_data")]
        if tabled is not None:
            path_sets.append(tuple(tabled))
        for paths in path_sets:
            if not paths:
                raise pytest.UsageError(f"{item.nodeid}: repo_data needs at least one path")
            missing = repo_data_missing(tuple(str(arg) for arg in paths))
            if missing is not None:
                item.add_marker(pytest.mark.skip(reason=f"repo_data: {missing} is not in this checkout"))
                break
    run_live = os.environ.get("RUN_LIVE_LLM") == "1"
    run_local_e2e = os.environ.get("RUN_LOCAL_E2E") == "1"
    skip_live = pytest.mark.skip(reason=_LIVE_LLM_SKIP_REASON)
    skip_local_e2e = pytest.mark.skip(reason=_LOCAL_E2E_SKIP_REASON)
    for item in items:
        if "live_llm" in item.keywords and not run_live:
            item.add_marker(skip_live)
        if "local_e2e" in item.keywords and not run_local_e2e:
            item.add_marker(skip_local_e2e)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    return _build_workspace(tmp_path)


@pytest.fixture
def validation_workspace(tmp_path: Path):
    return make_validation_workspace(tmp_path)
