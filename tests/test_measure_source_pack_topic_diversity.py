"""Unit tests for `scripts/measure_source_pack_topic_diversity.py`.

The heart of this module (Phase 10.1.1.1, PROD-15, T-10.1.1.1-06) is proving
the diversity gate is genuinely falsifiable, not tautological: a matched pair
of tests (``test_diversity_gate_fails_when_every_deck_selects_the_same_papers``
and ``test_diversity_gate_passes_when_decks_select_distinct_papers``) drives
the script's own ``main(argv)`` entry point against the same baseline receipt
and asserts on the **process exit code**, not only on receipt fields -- a
metric that always exits 0 would manufacture false assurance.

Every fixture is a self-contained synthetic ``tmp_path`` workspace (narrative
source packs, a papers-store ``chunks.jsonl``, and ``matched_papers.json``
files) -- no live conference workspace data is required or read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_source_pack_topic_diversity.py"

_PROVIDER_SDK_MODULE_NAMES = ("anthropic", "openai")

RUN_ID = "test-run"

# 16 distinct 64-character lowercase hex ids, each a single repeated hex
# digit -- realistic in shape (the hex64 filter this script applies is part
# of what is being tested) without needing a real sha256 digest.
_HEX_DIGITS = "0123456789abcdef"


def _hex_id(n: int) -> str:
    return _HEX_DIGITS[n] * 64


def _load_module() -> Any:
    """Load the module under test, insulated from ambient sys.modules state.

    Mirrors tests/test_measure_blueprint_content_sufficiency.py's own
    fixture: the script under test deliberately raises at import time if
    `anthropic`/`openai` is already in `sys.modules`, so this loader
    temporarily hides (and always restores) any pre-existing entries.
    """

    spec = importlib.util.spec_from_file_location("measure_source_pack_topic_diversity_under_test", SCRIPT_PATH)
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_topic_pack(workspace: Path, topic_id: str, paper_hashes: list[str]) -> None:
    chunks = [
        {
            "chunk_id": f"paper:{paper_hash}:0",
            "extraction_record_id": f"paper:{paper_hash}",
            "quote": f"Selected evidence for {paper_hash[:8]}.",
        }
        for paper_hash in paper_hashes
    ]
    _write_json(workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json", {"chunks": chunks})


def _write_chunks_store(workspace: Path, run_id: str, entries: list[tuple[str, str]]) -> None:
    """``entries``: list of (paper_hash, conference_scope)."""

    path = workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for paper_hash, scope in entries:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": f"paper:{paper_hash}:0",
                        "artifact_id": f"paper:{paper_hash}",
                        "conference_scope": scope,
                        "text": f"Corpus evidence for {paper_hash[:8]}.",
                        "source_path": f"https://example.test/{paper_hash}",
                    }
                )
                + "\n"
            )


def _write_matched_papers(workspace: Path, run_id: str, topic_id: str, paper_ids: list[str]) -> None:
    _write_json(
        workspace / "storage" / "paper-matching" / run_id / topic_id / "matched_papers.json",
        {"papers": [{"paper_id": paper_id, "rank": index + 1} for index, paper_id in enumerate(paper_ids)]},
    )


def _write_baseline_receipt(path: Path, *, union_count: int, intersection_count: int) -> None:
    _write_json(path, {"diversity": {"union_count": union_count, "intersection_count": intersection_count}})


def _run_main(mod: Any, tmp_path: Path, workspace: Path, *, extra_args: list[str] | None = None) -> tuple[int, dict[str, Any]]:
    output_receipt = tmp_path / "receipt.json"
    argv = [
        "--workspace",
        str(workspace),
        "--run-id",
        RUN_ID,
        "--output-receipt",
        str(output_receipt),
        "--json",
    ] + list(extra_args or [])
    exit_code = mod.main(argv)
    receipt = json.loads(output_receipt.read_text(encoding="utf-8"))
    return exit_code, receipt


def _topic_entry(receipt: dict[str, Any], topic_id: str) -> dict[str, Any]:
    matches = [topic for topic in receipt["topics"] if topic["topic_id"] == topic_id]
    assert matches, f"topic {topic_id} not found in receipt topics: {receipt['topics']}"
    return matches[0]


# --- The non-tautology proof: a matched pass/fail pair against one baseline ---


def test_diversity_gate_fails_when_every_deck_selects_the_same_papers(mod: Any, tmp_path: Path) -> None:
    """Three decks all select the identical 5 main-conference papers.

    Measured union/intersection reproduce the baseline exactly (no
    improvement), so `--fail-under-baseline` must exit 1 -- proving the
    gate is capable of failing on degenerate data, not merely capable of
    passing on any input.
    """

    workspace = tmp_path / "workspace"
    shared_ids = [_hex_id(i) for i in range(5)]
    for topic_id in ("topic_a", "topic_b", "topic_c"):
        _write_topic_pack(workspace, topic_id, shared_ids)
    _write_chunks_store(workspace, RUN_ID, [(paper_hash, "main_conference") for paper_hash in shared_ids])

    baseline_path = tmp_path / "baseline.json"
    _write_baseline_receipt(baseline_path, union_count=5, intersection_count=5)

    exit_code, receipt = _run_main(
        mod,
        tmp_path,
        workspace,
        extra_args=["--baseline-receipt", str(baseline_path), "--fail-under-baseline"],
    )

    assert receipt["diversity"]["union_count"] == 5
    assert receipt["diversity"]["intersection_count"] == 5
    assert receipt["diversity_improved_vs_baseline"] is False
    assert exit_code == 1


def test_diversity_gate_passes_when_decks_select_distinct_papers(mod: Any, tmp_path: Path) -> None:
    """The same three decks, now selecting disjoint main-conference papers.

    The only difference from the failing test above is the measured data --
    same baseline, same shape, same flags. Union grows past the baseline and
    intersection collapses to 0, so the gate exits 0.
    """

    workspace = tmp_path / "workspace"
    topic_ids_by_offset = {"topic_a": 0, "topic_b": 5, "topic_c": 10}
    all_entries: list[tuple[str, str]] = []
    for topic_id, offset in topic_ids_by_offset.items():
        ids = [_hex_id(offset + i) for i in range(5)]
        _write_topic_pack(workspace, topic_id, ids)
        all_entries.extend((paper_hash, "main_conference") for paper_hash in ids)
    _write_chunks_store(workspace, RUN_ID, all_entries)

    baseline_path = tmp_path / "baseline.json"
    _write_baseline_receipt(baseline_path, union_count=5, intersection_count=5)

    exit_code, receipt = _run_main(
        mod,
        tmp_path,
        workspace,
        extra_args=["--baseline-receipt", str(baseline_path), "--fail-under-baseline"],
    )

    assert receipt["diversity"]["union_count"] == 15
    assert receipt["diversity"]["union_count"] > receipt["baseline_union_count"]
    assert receipt["diversity"]["intersection_count"] == 0
    assert receipt["diversity"]["intersection_count"] < receipt["baseline_intersection_count"]
    assert receipt["diversity_improved_vs_baseline"] is True
    assert exit_code == 0


# --- Provenance split, zero-joinable reporting, scope filtering, tolerance ---


def test_provenance_split_labels_priority_versus_keyword_backfill(mod: Any, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ids = [_hex_id(0), _hex_id(1), _hex_id(2)]
    _write_topic_pack(workspace, "topic_x", ids)
    _write_chunks_store(workspace, RUN_ID, [(paper_hash, "main_conference") for paper_hash in ids])
    _write_matched_papers(workspace, RUN_ID, "topic_x", [ids[0]])

    _, receipt = _run_main(mod, tmp_path, workspace)
    topic = _topic_entry(receipt, "topic_x")

    assert topic["priority_source_ids"] == [ids[0]]
    assert sorted(topic["keyword_backfill_source_ids"]) == sorted(ids[1:])
    assert set(topic["priority_source_ids"]).isdisjoint(topic["keyword_backfill_source_ids"])
    assert sorted(topic["priority_source_ids"] + topic["keyword_backfill_source_ids"]) == sorted(
        topic["main_conference_source_ids"]
    )


def test_topic_with_only_non_hex_matched_paper_ids_is_reported_not_dropped(mod: Any, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ids = [_hex_id(3), _hex_id(4)]
    _write_topic_pack(workspace, "topic_y", ids)
    _write_chunks_store(workspace, RUN_ID, [(paper_hash, "main_conference") for paper_hash in ids])
    _write_matched_papers(workspace, RUN_ID, "topic_y", ["arXiv:2504.10479", "openreview-slug-abc123"])

    _, receipt = _run_main(mod, tmp_path, workspace)
    topic = _topic_entry(receipt, "topic_y")

    assert topic["joinable_matched_paper_count"] == 0
    assert sorted(topic["main_conference_source_ids"]) == sorted(ids)
    assert set(ids).issubset(set(receipt["diversity"]["union_source_ids"]))
    assert "topic_y" in receipt["diversity"]["topics_entirely_backfilled"]


def test_workshop_scoped_selections_are_excluded_from_the_metric(mod: Any, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workshop_id = _hex_id(5)
    main_ids = [_hex_id(6), _hex_id(7)]
    _write_topic_pack(workspace, "topic_z", [workshop_id, *main_ids])
    _write_chunks_store(
        workspace,
        RUN_ID,
        [(workshop_id, "workshop"), (main_ids[0], "main_conference"), (main_ids[1], "main_conference")],
    )

    _, receipt = _run_main(mod, tmp_path, workspace)
    topic = _topic_entry(receipt, "topic_z")

    assert topic["main_conference_source_ids"] == sorted(main_ids)
    assert topic["main_conference_source_count"] == 2
    assert workshop_id not in topic["main_conference_source_ids"]


def test_missing_matched_papers_file_is_tolerated(mod: Any, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ids = [_hex_id(8), _hex_id(9)]
    _write_topic_pack(workspace, "topic_w", ids)
    _write_chunks_store(workspace, RUN_ID, [(paper_hash, "main_conference") for paper_hash in ids])
    # Deliberately no matched_papers.json anywhere for topic_w.

    exit_code, receipt = _run_main(mod, tmp_path, workspace)
    topic = _topic_entry(receipt, "topic_w")

    assert topic["joinable_matched_paper_count"] == 0
    assert topic["main_conference_source_count"] == 2
    assert exit_code == 0
