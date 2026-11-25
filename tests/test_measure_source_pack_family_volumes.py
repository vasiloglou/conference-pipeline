"""Unit tests for `scripts/measure_source_pack_family_volumes.py`.

These tests exercise the pure family-resolution and per-store counting logic
against synthetic chunk dicts only -- no live CVPR2026 workspace is required
(Task 1's `<action>` explicitly calls this out). The script's own runner-
loading/CLI path (which does need a live workspace) is exercised separately
by Task 2's live measurement run, not by this unit-test module.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_source_pack_family_volumes.py"

_PROVIDER_SDK_MODULE_NAMES = ("anthropic", "openai")


def _load_module() -> Any:
    """Load the module under test, insulated from ambient sys.modules state.

    The script under test deliberately raises at import time if
    ``anthropic``/``openai`` is already in ``sys.modules`` (a fail-closed
    guard this file has its own dedicated test for, below). Run as part of
    the full repo test suite -- rather than in isolation -- some earlier,
    unrelated test file legitimately imports one of those SDKs first, which
    would otherwise make this fixture's outcome depend on suite-wide test
    order instead of on this module's own behavior. Temporarily hiding (and
    always restoring) any pre-existing entries keeps this load exercising
    only the fresh-process scenario the guard is actually meant to protect
    -- a real CLI invocation, never a shared, already-polluted process --
    without weakening the guard itself in any way.
    """

    spec = importlib.util.spec_from_file_location(
        "measure_source_pack_family_volumes_under_test", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    saved_provider_modules = {
        name: sys.modules.pop(name)
        for name in _PROVIDER_SDK_MODULE_NAMES
        if name in sys.modules
    }
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.update(saved_provider_modules)
    return module


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load_module()


def _trivial_compact(chunk: dict[str, Any]) -> dict[str, Any]:
    """Minimal stand-in for the runner's `_analysis_compact_chunk`.

    Only supplies the two fields the counting logic actually reads
    (`extraction_record_id`, `quote`), so these tests never need to load the
    full CVPR2026 runner module or any live workspace.
    """

    return {
        "extraction_record_id": str(chunk.get("artifact_id") or ""),
        "quote": str(chunk.get("text") or ""),
    }


# --- Behavior block: resolve_chunk_family --------------------------------


def test_resolve_chunk_family_top_level_workshop(mod: Any) -> None:
    chunk = {"conference_scope": "workshop"}
    assert mod.resolve_chunk_family(chunk) == "workshop"


def test_resolve_chunk_family_falls_back_to_source_context(mod: Any) -> None:
    chunk = {"source_context": {"conference_scope": "workshop"}}
    assert mod.resolve_chunk_family(chunk) == "workshop"


def test_resolve_chunk_family_fails_open_when_missing(mod: Any) -> None:
    chunk: dict[str, Any] = {}
    assert mod.resolve_chunk_family(chunk) == "main_conference"


def test_resolve_chunk_family_fails_open_when_unrecognized(mod: Any) -> None:
    chunk = {"conference_scope": "not_a_real_family"}
    assert mod.resolve_chunk_family(chunk) == "main_conference"


def test_resolve_chunk_family_top_level_takes_priority_over_source_context(
    mod: Any,
) -> None:
    chunk = {
        "conference_scope": "main_conference",
        "source_context": {"conference_scope": "workshop"},
    }
    assert mod.resolve_chunk_family(chunk) == "main_conference"


def test_resolve_chunk_family_unrecognized_source_context_fails_open(mod: Any) -> None:
    chunk = {"source_context": {"conference_scope": "bogus"}}
    assert mod.resolve_chunk_family(chunk) == "main_conference"


# --- Behavior block: distinct-source de-dup ------------------------------


def test_distinct_source_count_dedups_shared_artifact_id(mod: Any) -> None:
    topic_specs = ({"id": "topic_a", "title": "Topic A", "keywords": ("vision",)},)
    chunks = [
        {
            "artifact_id": "paper-1",
            "text": "vision paper evidence chunk one",
            "conference_scope": "main_conference",
        },
        {
            "artifact_id": "paper-1",
            "text": "vision paper evidence chunk two",
            "conference_scope": "main_conference",
        },
    ]
    result = mod.measure_store_family_volumes(
        chunks, topic_specs=topic_specs, compact_chunk_fn=_trivial_compact
    )
    stats = result["per_topic"]["topic_a"]["main_conference"]
    assert stats["matching_chunk_count"] == 2
    assert stats["distinct_source_count"] == 1


# --- Denominator integrity ------------------------------------------------


def test_family_chunk_counts_sum_to_total_including_unclassified(mod: Any) -> None:
    topic_specs = ({"id": "topic_a", "title": "Topic A", "keywords": ("vision",)},)
    chunks = [
        {"artifact_id": "p1", "text": "vision", "conference_scope": "workshop"},
        {"artifact_id": "p2", "text": "vision", "conference_scope": "main_conference"},
        # Unrecognized conference_scope: fails open to main_conference but is
        # also tracked in raw_unclassified_count so it can't be hidden by a
        # shrinking denominator.
        {"artifact_id": "p3", "text": "no keyword here", "conference_scope": "bogus"},
    ]
    result = mod.measure_store_family_volumes(
        chunks, topic_specs=topic_specs, compact_chunk_fn=_trivial_compact
    )
    assert result["total_chunk_count"] == 3
    assert sum(result["family_chunk_counts"].values()) == result["total_chunk_count"]
    assert result["family_chunk_counts"]["workshop"] == 1
    assert result["family_chunk_counts"]["main_conference"] == 2
    assert result["raw_unclassified_count"] == 1


def test_first_match_chunk_index_records_ordering_evidence(mod: Any) -> None:
    """The direct, measured expression of Finding D: workshop matches are
    reachable at a lower file offset than main-conference matches."""

    topic_specs = ({"id": "topic_a", "title": "Topic A", "keywords": ("vision",)},)
    chunks = [
        {"artifact_id": "w1", "text": "vision workshop chunk", "conference_scope": "workshop"},
        {"artifact_id": "m1", "text": "vision main chunk", "conference_scope": "main_conference"},
    ]
    result = mod.measure_store_family_volumes(
        chunks, topic_specs=topic_specs, compact_chunk_fn=_trivial_compact
    )
    stats = result["per_topic"]["topic_a"]
    assert stats["workshop"]["first_match_chunk_index"] == 0
    assert stats["main_conference"]["first_match_chunk_index"] == 1


def test_first_match_chunk_index_is_none_when_family_never_matches(mod: Any) -> None:
    topic_specs = ({"id": "topic_a", "title": "Topic A", "keywords": ("vision",)},)
    chunks = [
        {"artifact_id": "w1", "text": "vision workshop chunk", "conference_scope": "workshop"},
    ]
    result = mod.measure_store_family_volumes(
        chunks, topic_specs=topic_specs, compact_chunk_fn=_trivial_compact
    )
    stats = result["per_topic"]["topic_a"]
    assert stats["workshop"]["first_match_chunk_index"] == 0
    assert stats["main_conference"]["first_match_chunk_index"] is None
    assert stats["main_conference"]["matching_chunk_count"] == 0
    assert stats["main_conference"]["distinct_source_count"] == 0


# --- Module-level shape / governance guards -------------------------------


def test_conference_scope_families_constant(mod: Any) -> None:
    assert mod.CONFERENCE_SCOPE_FAMILIES == ("main_conference", "workshop")


def test_no_provider_sdk_imports_in_module_source() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "import anthropic" not in stripped
        assert "import openai" not in stripped


@pytest.mark.parametrize("provider_module_name", _PROVIDER_SDK_MODULE_NAMES)
def test_module_load_fails_closed_when_provider_sdk_preimported(
    provider_module_name: str,
) -> None:
    """Direct test of the fail-closed guard itself: loading the module while
    a provider SDK is already in sys.modules must raise, not silently run.

    Pops *both* provider names first (not just the one under test) so this
    test's outcome is deterministic regardless of which provider SDK, if
    any, an earlier test in the same pytest session already imported --
    otherwise, e.g., an ambient "anthropic" import would make the "openai"
    parametrization observe the guard's anthropic message instead.
    """

    saved_provider_modules = {
        name: sys.modules.pop(name)
        for name in _PROVIDER_SDK_MODULE_NAMES
        if name in sys.modules
    }
    sys.modules[provider_module_name] = types.ModuleType(provider_module_name)
    try:
        spec = importlib.util.spec_from_file_location(
            "measure_source_pack_family_volumes_guard_check", SCRIPT_PATH
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(RuntimeError, match=provider_module_name):
            spec.loader.exec_module(module)
    finally:
        del sys.modules[provider_module_name]
        sys.modules.update(saved_provider_modules)


def test_cli_help_lists_required_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for flag in ("--workspace", "--run-id", "--output-receipt", "--json"):
        assert flag in result.stdout
