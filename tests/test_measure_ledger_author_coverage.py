"""Unit tests for `scripts/measure_ledger_author_coverage.py`.

These tests exercise the pure per-family bucket-resolution and counting logic
against synthetic ``speaker_author_work_ledger.json``-shaped entry dicts only
-- no live workspace is required. The script's CLI path (workspace walk +
receipt write) is exercised separately by Task 3's live measurement run
against ``conferences/cvpr2026``, not by this unit-test module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_ledger_author_coverage.py"

_PROVIDER_SDK_MODULE_NAMES = ("anthropic", "openai")


def _load_module() -> Any:
    """Load the module under test, insulated from ambient sys.modules state.

    Mirrors tests/test_measure_source_pack_family_volumes.py's own fixture:
    the script under test deliberately raises at import time if
    ``anthropic``/``openai`` is already in ``sys.modules``, so this loader
    temporarily hides (and always restores) any pre-existing entries -- the
    outcome must depend only on this module's own behavior, never on
    whichever unrelated test ran earlier in the shared pytest process.
    """

    spec = importlib.util.spec_from_file_location("measure_ledger_author_coverage_under_test", SCRIPT_PATH)
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


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "source_id": "paper_abstract:example",
        "resolved_title": "",
        "authors": [],
        "speakers": [],
        "source_family": "papers",
        "evidence_paths": ["analysis/topic_reference_ledgers/example.json"],
        "status": "unresolved",
    }
    base.update(overrides)
    return base


# --- Fail-closed provider-SDK guard ---------------------------------------


def test_module_load_fails_closed_when_provider_sdk_preimported() -> None:
    fake_anthropic = type(sys)("anthropic")
    saved = sys.modules.get("anthropic")
    sys.modules["anthropic"] = fake_anthropic
    try:
        spec = importlib.util.spec_from_file_location("measure_ledger_author_coverage_guard_probe", SCRIPT_PATH)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(RuntimeError):
            spec.loader.exec_module(module)
    finally:
        if saved is not None:
            sys.modules["anthropic"] = saved
        else:
            sys.modules.pop("anthropic", None)


# --- Behavior block bullet 1: author-bearing count vs. total --------------


def test_author_bearing_count_vs_total_for_one_family(mod: Any) -> None:
    entries = [
        _entry(source_id="paper_abstract:a", authors=["Ada Lovelace"], evidence_paths=["storage/download-corpus/artifacts/paper_pdf/a.pdf"]),
        _entry(source_id="paper_abstract:b", authors=[], evidence_paths=["storage/download-corpus/artifacts/paper_pdf/b.pdf"]),
    ]
    buckets = mod.measure_ledger(entries)
    main = buckets[mod.BUCKET_MAIN_CONFERENCE_PAPERS]
    assert main["total"] == 2
    assert main["authors_present"] == 1


# --- Behavior block bullet 2: bucketing + explicit unclassified -----------


def test_artifacts_family_buckets_as_artifacts(mod: Any) -> None:
    entries = [_entry(source_id="speech_transcript:x", source_family="artifacts", evidence_paths=["storage/download-corpus/event-media/captions/x.en.vtt"])]
    buckets = mod.measure_ledger(entries)
    assert buckets[mod.BUCKET_ARTIFACTS]["total"] == 1
    assert buckets[mod.BUCKET_MAIN_CONFERENCE_PAPERS]["total"] == 0


def test_papers_family_with_workshop_path_buckets_as_workshop(mod: Any) -> None:
    entries = [
        _entry(
            source_id="paper_abstract:thecvf.com_CVPR_2026_Workshop_3D4S_1FdAbS4FoQ",
            authors=["Sophia Zorek"],
            evidence_paths=[
                "analysis/topic_reference_ledgers/example.json",
                "storage/workshops/papers/thecvf-com_cvpr_2026_workshop_3d4s/1FdAbS4FoQ.pdf",
            ],
        )
    ]
    buckets = mod.measure_ledger(entries)
    assert buckets[mod.BUCKET_WORKSHOP_PAPERS]["total"] == 1
    assert buckets[mod.BUCKET_WORKSHOP_PAPERS]["authors_present"] == 1
    assert buckets[mod.BUCKET_MAIN_CONFERENCE_PAPERS]["total"] == 0


def test_papers_family_with_real_non_workshop_path_buckets_as_main_conference(mod: Any) -> None:
    entries = [
        _entry(
            source_id="paper_abstract:realhash",
            authors=["Xiaofan Que"],
            evidence_paths=[
                "analysis/topic_reference_ledgers/example.json",
                "storage/download-corpus/artifacts/paper_pdf/realhash.pdf",
            ],
        )
    ]
    buckets = mod.measure_ledger(entries)
    assert buckets[mod.BUCKET_MAIN_CONFERENCE_PAPERS]["total"] == 1
    assert buckets[mod.BUCKET_MAIN_CONFERENCE_PAPERS]["authors_present"] == 1
    assert buckets[mod.BUCKET_WORKSHOP_PAPERS]["total"] == 0


def test_papers_family_with_no_real_evidence_path_is_unclassified_not_dropped(mod: Any) -> None:
    """A matched-paper entry (title cross-reference, no chunk evidence) carries
    only the generic ledger self-reference path -- no positive main-vs-workshop
    signal -- so it must land in `unclassified`, never silently assumed main."""

    entries = [
        _entry(
            source_id="paper_abstract:arXiv_2307.16125",
            authors=[],
            evidence_paths=["analysis/topic_reference_ledgers/example.json"],
        )
    ]
    buckets = mod.measure_ledger(entries)
    assert buckets[mod.BUCKET_UNCLASSIFIED]["total"] == 1
    assert buckets[mod.BUCKET_MAIN_CONFERENCE_PAPERS]["total"] == 0
    assert buckets[mod.BUCKET_WORKSHOP_PAPERS]["total"] == 0


def test_unrecognized_source_family_is_unclassified_not_dropped(mod: Any) -> None:
    entries = [_entry(source_family="mystery", evidence_paths=[])]
    buckets = mod.measure_ledger(entries)
    assert buckets[mod.BUCKET_UNCLASSIFIED]["total"] == 1


def test_bucket_denominators_sum_to_total_entry_count(mod: Any) -> None:
    entries = [
        _entry(source_family="artifacts", evidence_paths=["storage/download-corpus/event-media/captions/x.en.vtt"]),
        _entry(source_family="papers", evidence_paths=["storage/workshops/papers/a/b.pdf"]),
        _entry(source_family="papers", evidence_paths=["storage/download-corpus/artifacts/paper_pdf/c.pdf"]),
        _entry(source_family="papers", evidence_paths=["analysis/topic_reference_ledgers/example.json"]),
        _entry(source_family="mystery", evidence_paths=[]),
    ]
    buckets = mod.measure_ledger(entries)
    assert sum(bucket["total"] for bucket in buckets.values()) == len(entries)


# --- Behavior block bullet 3: placeholder person names --------------------


@pytest.mark.parametrize("placeholder", ["Unknown", "N/A", "anonymous", "  Unnamed  ", ""])
def test_placeholder_person_name_is_not_a_resolved_author(mod: Any, placeholder: str) -> None:
    entries = [
        _entry(
            authors=[placeholder],
            evidence_paths=["storage/download-corpus/artifacts/paper_pdf/x.pdf"],
        )
    ]
    buckets = mod.measure_ledger(entries)
    assert buckets[mod.BUCKET_MAIN_CONFERENCE_PAPERS]["authors_present"] == 0


def test_placeholder_speaker_name_alone_does_not_count_but_real_speaker_does(mod: Any) -> None:
    entries = [
        _entry(source_family="artifacts", authors=[], speakers=["unknown"], evidence_paths=["storage/download-corpus/event-media/captions/x.en.vtt"]),
        _entry(source_family="artifacts", authors=[], speakers=["Fei-Fei Li"], evidence_paths=["storage/download-corpus/event-media/captions/y.en.vtt"]),
    ]
    buckets = mod.measure_ledger(entries)
    assert buckets[mod.BUCKET_ARTIFACTS]["total"] == 2
    assert buckets[mod.BUCKET_ARTIFACTS]["authors_present"] == 1


# --- Behavior block bullet 4: no blended total in the receipt -------------


def test_receipt_reports_buckets_separately_with_no_blended_key(mod: Any, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    deck_dir = workspace / "decks" / "01_topic"
    deck_dir.mkdir(parents=True)
    ledger_payload = {
        "entries": [
            _entry(source_family="artifacts", evidence_paths=["storage/download-corpus/event-media/captions/x.en.vtt"]),
            _entry(source_family="papers", authors=["A. Author"], evidence_paths=["storage/download-corpus/artifacts/paper_pdf/c.pdf"]),
        ]
    }
    (deck_dir / "speaker_author_work_ledger.json").write_text(json.dumps(ledger_payload), encoding="utf-8")

    result = mod.measure_workspace(workspace)
    assert set(result["workspace_totals"].keys()) == set(mod.BUCKETS)
    for forbidden_key in ("blended", "overall_coverage", "total_authors_coverage_pct"):
        assert forbidden_key not in result

    receipt_path = tmp_path / "receipt.json"
    receipt = mod.write_coverage_receipt(
        receipt_path,
        workspace=workspace,
        run_id="test-run",
        measurement=result,
    )
    assert receipt["provider_calls_made"] is False
    assert receipt["live_provider_calls_made"] is False
    assert set(receipt["workspace_totals"].keys()) == set(mod.BUCKETS)
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


# --- CLI surface ------------------------------------------------------------


def test_cli_help_exits_zero(mod: Any) -> None:
    with pytest.raises(SystemExit) as excinfo:
        mod.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_no_floor_or_threshold_identifiers_defined() -> None:
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    stripped = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    for forbidden in ("MIN_SOURCES", "MIN_PEOPLE", "COVERAGE_FLOORS"):
        assert forbidden not in stripped
