"""Unit tests for `scripts/measure_blueprint_content_sufficiency.py`.

These tests exercise `measure_deck` against synthetic
`speaker_author_work_ledger.json`-shaped ledgers and the real, live
`scripts/presentations/generate_editorial_blueprint.py` module (loaded
read-only via `load_blueprint_module`) -- never a mocked/faked catalog
function, since the whole point of this script is to measure against the
actual live `build_source_catalog` and the actual live threshold values, not
a stand-in for them. No provider SDK is imported and no live workspace is
required; the script's CLI/workspace-walk path is exercised separately by
Task 1's live measurement run against `conferences/cvpr2026`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_blueprint_content_sufficiency.py"

_PROVIDER_SDK_MODULE_NAMES = ("anthropic", "openai")


def _load_module() -> Any:
    """Load the module under test, insulated from ambient sys.modules state.

    Mirrors tests/test_measure_ledger_author_coverage.py's own fixture: the
    script under test deliberately raises at import time if
    `anthropic`/`openai` is already in `sys.modules`, so this loader
    temporarily hides (and always restores) any pre-existing entries -- the
    outcome must depend only on this module's own behavior, never on
    whichever unrelated test ran earlier in the shared pytest process.
    """

    spec = importlib.util.spec_from_file_location("measure_blueprint_content_sufficiency_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    saved_provider_modules = {name: sys.modules.pop(name) for name in _PROVIDER_SDK_MODULE_NAMES if name in sys.modules}
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.update(saved_provider_modules)
    return module


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load_module()


@pytest.fixture(scope="module")
def blueprint_module(mod: Any) -> Any:
    """Load the real, live generate_editorial_blueprint.py module, read-only."""

    return mod.load_blueprint_module()


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "source_id": "paper_abstract:example",
        "resolved_title": "Example Paper Title",
        "authors": ["Jane Doe", "John Smith"],
        "speakers": [],
        "source_family": "papers",
        "visible_source_chip_eligible": True,
        "contribution_note": "Discusses example methodology.",
        "resolution_status": "resolved",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


# --- Fail-closed provider-SDK guard ---------------------------------------


def test_module_load_fails_closed_when_provider_sdk_preimported() -> None:
    fake_anthropic = type(sys)("anthropic")
    saved = sys.modules.get("anthropic")
    sys.modules["anthropic"] = fake_anthropic
    try:
        spec = importlib.util.spec_from_file_location(
            "measure_blueprint_content_sufficiency_guard_probe", SCRIPT_PATH
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(RuntimeError):
            spec.loader.exec_module(module)
    finally:
        if saved is not None:
            sys.modules["anthropic"] = saved
        else:
            sys.modules.pop("anthropic", None)


# --- Behavior block coverage ------------------------------------------------


def test_ineligible_ledger_yields_zero_eligible_sources(mod: Any, blueprint_module: Any) -> None:
    """Given a ledger whose entries are all ineligible for a visible source
    chip, the eligible source count is 0."""

    ledger = {
        "entries": [
            _entry(source_id="paper_abstract:a", visible_source_chip_eligible=False),
            _entry(source_id="paper_abstract:b", visible_source_chip_eligible=False),
        ]
    }
    result = mod.measure_deck(ledger, blueprint_module)
    assert result["eligible_source_count"] == 0


def test_shared_work_identity_deduplicates_to_distinct_works(mod: Any, blueprint_module: Any) -> None:
    """Given a ledger with three eligible entries sharing one work identity,
    build_source_catalog's own de-duplication is respected and the eligible
    count reflects distinct works, not raw entry count."""

    ledger = {
        "entries": [
            _entry(source_id="paper_abstract:same-work:1", resolved_title="Shared Work: Part One"),
            _entry(source_id="paper_abstract:same-work:2", resolved_title="Shared Work: Part Two"),
            _entry(source_id="paper_abstract:same-work:3", resolved_title="Shared Work: Part Three"),
        ]
    }
    result = mod.measure_deck(ledger, blueprint_module)
    # All three entries share the same base title ("shared work"), so
    # build_source_catalog's own _catalog_work_key collapses them to one
    # distinct work -- the eligible count must reflect 1, not 3.
    assert result["eligible_source_count"] == 1


def test_named_people_ceiling_is_union_of_catalog_people_with_upper_bound_qualifier(
    mod: Any, blueprint_module: Any
) -> None:
    """The named-people ceiling equals the size of the union of the catalog
    entries' people lists, and is reported alongside an explicit upper-bound
    qualifier field in the receipt."""

    ledger = {
        "entries": [
            _entry(
                source_id="paper_abstract:x",
                resolved_title="Distinct Paper X",
                authors=["Alice Alpha", "Bob Beta"],
            ),
            _entry(
                source_id="paper_abstract:y",
                resolved_title="Distinct Paper Y",
                authors=["Bob Beta", "Carol Gamma"],
            ),
        ]
    }
    result = mod.measure_deck(ledger, blueprint_module)
    # Union of {"Alice Alpha", "Bob Beta"} and {"Bob Beta", "Carol Gamma"} is
    # exactly 3 distinct people, not 4 (a naive sum would double-count Bob).
    assert result["named_people_ceiling"] == 3
    qualifier = result["named_people_ceiling_qualifier"]
    assert qualifier["is_upper_bound"] is True
    assert qualifier["not_a_pass"] is True
    assert "paid" in qualifier["explanation"].lower()


def test_thresholds_are_read_live_from_module_not_restated(mod: Any, blueprint_module: Any) -> None:
    """Thresholds are read from the live generator module at measurement
    time; the script contains no threshold integer literal of its own."""

    ledger = {"entries": []}
    result = mod.measure_deck(ledger, blueprint_module)
    assert result["min_sources"] == blueprint_module.MIN_SOURCES
    assert result["min_people"] == blueprint_module.MIN_PEOPLE

    # Grep the script's own source for a restated threshold literal -- this
    # mirrors the plan's own acceptance-criteria grep so a future edit that
    # accidentally hardcodes a value fails this test, not just the shell
    # command run manually at plan-execution time.
    script_lines = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    non_comment_lines = [line for line in script_lines if not line.strip().startswith("#")]
    joined = "\n".join(non_comment_lines)
    for restated_literal in (str(blueprint_module.MIN_SOURCES), str(blueprint_module.MIN_PEOPLE)):
        import re

        assert not re.search(rf"(?<!\d){re.escape(restated_literal)}(?!\d)", joined), (
            f"threshold literal {restated_literal!r} must not be restated in the script"
        )


def test_eligible_source_count_pass_and_named_people_clears_are_boolean_gates(
    mod: Any, blueprint_module: Any
) -> None:
    """Pass/clear indicators are booleans computed against the live thresholds,
    not hardcoded True/False."""

    ledger = {"entries": []}
    result = mod.measure_deck(ledger, blueprint_module)
    assert result["eligible_source_count_pass"] is False
    assert result["named_people_ceiling_clears_min_people"] is False
    assert isinstance(result["eligible_source_count_pass"], bool)
    assert isinstance(result["named_people_ceiling_clears_min_people"], bool)
