"""Tests for the shared, order-independent source-pack chunk selector.

This module encodes Finding D's contract (Phase 10.1.1, PROD-15): the
per-topic narrative source-pack scan over the ``papers`` vector-store
sidecar must not depend on the *physical file order* of
``main_conference``-scoped versus ``workshop``-scoped chunks. Before this
fix, a single shared per-topic bucket capped at
``ANALYSIS_SOURCE_PACK_MAX_CHUNKS`` would fill entirely from whichever
family the scanner reached first -- on the live CVPR2026 papers store,
workshop chunks occupy indices 0-7180 and main_conference chunks start at
index 7181 (measured in
``.planning/workstreams/cvpr-2026/phases/10.1.1-source-pack-family-stratification/10.1.1-FAMILY-VOLUMES.md``),
so a cap-then-stop scan reached its 8-chunk cap from workshop matches alone
7,181 chunk-positions before it ever saw a main_conference match.

The fix stratifies the ``papers`` store into per-family sub-buckets with a
floor-with-backfill composition: a minimum guarantee for main_conference
representation, never a hard reservation that could leave slots
permanently empty for a topic with no workshop matches (or vice versa).
"""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.analysis.source_pack_selection import (
    DEFAULT_MAIN_CONFERENCE_FLOOR,
    DEFAULT_MAX_CHUNKS_PER_TOPIC,
    DEFAULT_PAPER_TARGET,
    SourcePackSelection,
    chunk_conference_scope,
    select_topic_source_chunks,
)


def _compact_chunk(chunk: dict[str, object]) -> dict[str, object]:
    """Mirror ``scripts/run_cvpr2026_production.py``'s ``_analysis_compact_chunk``
    shape closely enough to exercise the selector realistically, without
    pulling in that script's provider-SDK imports."""

    text = " ".join(str(chunk.get("text") or "").split())
    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "extraction_record_id": str(chunk.get("artifact_id") or ""),
        "source_artifact_path": str(chunk.get("source_path") or ""),
        "quote": text[:700],
    }


def _write_chunks_jsonl(path: Path, chunks: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")


_TOPIC_SPECS = ({"id": "01_topic", "title": "Topic", "keywords": ("vision",)},)


def _workshop_chunk(index: int) -> dict[str, object]:
    return {
        "chunk_id": f"workshop:{index}:0",
        "artifact_id": f"workshop-paper-{index}",
        "source_path": f"storage/workshops/paper-{index}.pdf",
        "text": "Vision workshop evidence for the shared topic.",
        "conference_scope": "workshop",
    }


def _main_chunk(index: int) -> dict[str, object]:
    return {
        "chunk_id": f"main:{index}:0",
        "artifact_id": f"main-paper-{index}",
        "source_path": f"https://cvpr.thecvf.com/paper-{index}",
        "text": "Vision main-conference evidence for the shared topic.",
        "conference_scope": "main_conference",
    }


def test_order_independence_workshop_first_still_yields_main_conference_representation(
    tmp_path: Path,
) -> None:
    """Finding D drop-proof.

    10.1.1-FAMILY-VOLUMES.md measured that on the live CVPR2026 papers
    store, EVERY one of the 12 topics' first workshop match occurs at chunk
    index 0 and first main_conference match occurs at chunk index 7181 --
    workshop chunks physically precede main_conference chunks by over 7,000
    positions. This fixture places 12 keyword-matching workshop chunks
    (more than ``DEFAULT_MAX_CHUNKS_PER_TOPIC``, so workshop alone could
    exhaust the cap) before 12 keyword-matching main_conference chunks, and
    asserts the composed pack still contains at least
    ``DEFAULT_MAIN_CONFERENCE_FLOOR`` main-conference chunks -- proven, per
    this plan's drop-style requirement, to FAIL against the pre-fix
    single-bucket selection body (see the plan SUMMARY for the recorded
    failure output).
    """

    store_dir = tmp_path / "papers"
    workshop_chunks = [_workshop_chunk(i) for i in range(12)]
    main_chunks = [_main_chunk(i) for i in range(12)]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", workshop_chunks + main_chunks)

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
    )

    composed = result.selected["01_topic"]
    main_ids = [c["chunk_id"] for c in composed if str(c["chunk_id"]).startswith("main:")]
    assert len(main_ids) >= DEFAULT_MAIN_CONFERENCE_FLOOR, (
        f"expected at least {DEFAULT_MAIN_CONFERENCE_FLOOR} main-conference chunks "
        f"despite workshop chunks occupying every earlier file position, got "
        f"{len(main_ids)}: {[c['chunk_id'] for c in composed]}"
    )


def test_reversed_file_order_yields_identical_selection(tmp_path: Path) -> None:
    """Selection is a function of chunk content, not physical file position:
    the same chunk set, written workshop-then-main or main-then-workshop,
    must select the identical chunk-id set. Family sizes are chosen to
    exactly match what floor-with-backfill composition needs (no family has
    more candidates than slots available to it), so no truncation-order
    ambiguity can hide behind this assertion."""

    max_chunks_per_topic = 6
    paper_target = 4
    main_conference_floor = 2

    main_chunks = [_main_chunk(i) for i in range(main_conference_floor)]
    workshop_chunks = [_workshop_chunk(i) for i in range(paper_target - main_conference_floor)]

    def run(chunks: list[dict[str, object]], subdir: str) -> set[str]:
        store_dir = tmp_path / subdir
        _write_chunks_jsonl(store_dir / "chunks.jsonl", chunks)
        result = select_topic_source_chunks(
            store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
            topic_specs=_TOPIC_SPECS,
            compact_chunk=_compact_chunk,
            max_chunks_per_topic=max_chunks_per_topic,
            paper_target=paper_target,
            main_conference_floor=main_conference_floor,
        )
        return {str(c["chunk_id"]) for c in result.selected["01_topic"]}

    workshop_first = run(workshop_chunks + main_chunks, "workshop-first")
    main_first = run(main_chunks + workshop_chunks, "main-first")

    assert workshop_first == main_first
    all_chunk_ids = {str(c["chunk_id"]) for c in (workshop_chunks + main_chunks)}
    assert workshop_first == all_chunk_ids


def test_floor_with_backfill_fills_full_quota_when_no_workshop_present(tmp_path: Path) -> None:
    """A topic with zero workshop matches must still fill its entire quota
    from main_conference chunks -- the floor is a minimum guarantee, never
    a hard reservation that leaves slots permanently empty."""

    store_dir = tmp_path / "papers"
    main_chunks = [_main_chunk(i) for i in range(10)]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", main_chunks)

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
    )

    composed = result.selected["01_topic"]
    assert len(composed) == DEFAULT_MAX_CHUNKS_PER_TOPIC, (
        f"expected the full quota ({DEFAULT_MAX_CHUNKS_PER_TOPIC}) to be filled from "
        f"main_conference alone, got {len(composed)}: {composed}"
    )
    assert result.family_counts["01_topic"]["workshop"] == 0
    assert result.family_counts["01_topic"]["main_conference"] == DEFAULT_MAX_CHUNKS_PER_TOPIC


def test_floor_with_backfill_fills_full_quota_when_no_main_conference_present(
    tmp_path: Path,
) -> None:
    """A topic with zero main_conference matches must still fill its entire
    quota from workshop chunks -- symmetric with the no-workshop case."""

    store_dir = tmp_path / "papers"
    workshop_chunks = [_workshop_chunk(i) for i in range(10)]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", workshop_chunks)

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
    )

    composed = result.selected["01_topic"]
    assert len(composed) == DEFAULT_MAX_CHUNKS_PER_TOPIC, (
        f"expected the full quota ({DEFAULT_MAX_CHUNKS_PER_TOPIC}) to be filled from "
        f"workshop alone, got {len(composed)}: {composed}"
    )
    assert result.family_counts["01_topic"]["main_conference"] == 0
    assert result.family_counts["01_topic"]["workshop"] == DEFAULT_MAX_CHUNKS_PER_TOPIC


def test_per_source_deduplication_survives_stratification(tmp_path: Path) -> None:
    """Many chunks sharing one artifact_id within one family contribute at
    most one chunk to that family's topic bucket -- the existing per-store
    seen-sources de-dup pattern extended one level deeper, per family."""

    store_dir = tmp_path / "papers"
    repeated_workshop_chunks = [
        {
            "chunk_id": f"workshop:same-paper:{i}",
            "artifact_id": "workshop-same-paper",
            "source_path": "storage/workshops/same-paper.pdf",
            "text": "Vision workshop evidence for the shared topic.",
            "conference_scope": "workshop",
        }
        for i in range(10)
    ]
    distinct_main_chunks = [_main_chunk(i) for i in range(5)]
    _write_chunks_jsonl(
        store_dir / "chunks.jsonl", repeated_workshop_chunks + distinct_main_chunks
    )

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
    )

    composed = result.selected["01_topic"]
    workshop_ids = [c["chunk_id"] for c in composed if str(c["chunk_id"]).startswith("workshop:")]
    assert len(workshop_ids) <= 1, (
        f"expected at most one chunk from the single shared workshop source, "
        f"got {workshop_ids}"
    )


def test_missing_conference_scope_fails_open_to_main_conference(tmp_path: Path) -> None:
    """A chunk with no conference_scope on either the record or its
    source_context is treated as main_conference and remains selectable --
    it must never be silently dropped."""

    store_dir = tmp_path / "papers"
    chunks = [
        {
            "chunk_id": f"unscoped:{i}:0",
            "artifact_id": f"unscoped-paper-{i}",
            "source_path": f"https://cvpr.thecvf.com/paper-{i}",
            "text": "Vision evidence with no conference_scope key at all.",
        }
        for i in range(5)
    ]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", chunks)

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
    )

    composed = result.selected["01_topic"]
    assert len(composed) == 5
    assert result.family_counts["01_topic"]["main_conference"] == 5
    assert result.family_counts["01_topic"]["workshop"] == 0


def test_cap_never_exceeds_default_and_equals_eight(tmp_path: Path) -> None:
    """The composed pack never exceeds DEFAULT_MAX_CHUNKS_PER_TOPIC, and
    that constant equals 8 -- raising it would mask order-dependence
    instead of removing it (ROADMAP done gate requires genuine
    order-independence, not a wider net)."""

    assert DEFAULT_MAX_CHUNKS_PER_TOPIC == 8
    assert DEFAULT_PAPER_TARGET <= DEFAULT_MAX_CHUNKS_PER_TOPIC
    assert DEFAULT_MAIN_CONFERENCE_FLOOR <= DEFAULT_PAPER_TARGET

    store_dir = tmp_path / "papers"
    abundant_chunks = [_main_chunk(i) for i in range(20)] + [
        _workshop_chunk(i) for i in range(20)
    ]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", abundant_chunks)

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
    )

    assert len(result.selected["01_topic"]) == DEFAULT_MAX_CHUNKS_PER_TOPIC


def test_chunk_conference_scope_reads_top_level_key() -> None:
    assert chunk_conference_scope({"conference_scope": "workshop"}) == "workshop"
    assert chunk_conference_scope({"conference_scope": "main_conference"}) == "main_conference"


def test_chunk_conference_scope_falls_back_to_source_context() -> None:
    chunk = {"source_context": {"conference_scope": "workshop"}}
    assert chunk_conference_scope(chunk) == "workshop"


def test_chunk_conference_scope_fails_open_for_missing_or_unrecognized_value() -> None:
    assert chunk_conference_scope({}) == "main_conference"
    assert chunk_conference_scope({"conference_scope": "typo-value"}) == "main_conference"
    assert chunk_conference_scope({"source_context": {}}) == "main_conference"


def test_source_pack_selection_dataclass_is_frozen() -> None:
    """Cheap contract test: SourcePackSelection is the documented frozen
    dataclass shape (selected, fallback, family_counts), so callers cannot
    accidentally mutate a shared selection result in place."""

    selection = SourcePackSelection(selected={}, fallback=[], family_counts={})
    try:
        selection.selected = {"x": []}  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("SourcePackSelection must be frozen")


# --- Phase 10.1.1.1 (PROD-15): content-derived main-conference ranking ---
#
# The tests below define the contract for the ranking that replaces file
# position: priority membership from the already-paid per-topic
# matched_papers.json signal, that signal's own ``rank`` field, then
# distinct-keyword-hit count -- with physical file position demoted to a
# last-resort determinism tiebreak only. Each fails against the pre-fix
# selector for a specific, named reason (see each docstring).


def _main_chunk_with_content(artifact_id: str, text: str) -> dict[str, object]:
    """Main-conference chunk with a caller-supplied ``artifact_id``/``text``,
    so keyword-hit-count and priority membership can be varied per chunk
    independently of file position. Mirrors ``_main_chunk``'s shape exactly;
    does not modify ``_main_chunk`` itself."""

    return {
        "chunk_id": f"main:{artifact_id}:0",
        "artifact_id": artifact_id,
        "source_path": f"https://cvpr.thecvf.com/{artifact_id}",
        "text": text,
        "conference_scope": "main_conference",
    }


_THREE_KEYWORD_TOPIC_SPECS = (
    {"id": "01_topic", "title": "Topic", "keywords": ("alpha", "beta", "gamma")},
)


def test_priority_source_ids_override_file_position(tmp_path: Path) -> None:
    """Drop-proof (Phase 10.1.1.1, PROD-15, Done Gate 1): a main-conference
    paper named in the topic's priority map must be selected even when its
    chunk is physically last in the file and would never be reached by an
    8-cap first-found scan. FAILS against the pre-fix selector, which stops
    scanning once the shared bucket reaches DEFAULT_MAX_CHUNKS_PER_TOPIC (8)
    -- chunks at indices 27-29 are never seen."""

    store_dir = tmp_path / "papers"
    chunks = [
        _main_chunk_with_content(
            f"main-paper-{i}", "Vision main-conference evidence for the shared topic."
        )
        for i in range(30)
    ]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", chunks)

    priority_source_ranks = {
        "01_topic": {
            "main-paper-27": 1,
            "main-paper-28": 2,
            "main-paper-29": 3,
        }
    }

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
        priority_source_ranks=priority_source_ranks,
    )

    composed_ids = {str(c["extraction_record_id"]) for c in result.selected["01_topic"]}
    for artifact_id in ("main-paper-27", "main-paper-28", "main-paper-29"):
        assert artifact_id in composed_ids, (
            f"expected priority-named paper {artifact_id!r} (physically last in the "
            f"file) to be selected; got {sorted(composed_ids)}"
        )


def test_priority_candidates_ranked_by_matched_papers_rank(tmp_path: Path) -> None:
    """matched_papers.json's own ``rank`` field breaks ties among
    more-than-floor priority candidates deterministically (Phase 10.1.1.1
    Q2). Twelve priority-named chunks are written in the REVERSE of their
    assigned rank order; the composed selection must still surface them in
    ascending-rank order, and the DEFAULT_MAIN_CONFERENCE_FLOOR best-ranked
    papers must all be present. FAILS against the pre-fix selector, which
    has no rank concept at all and would simply preserve file order."""

    store_dir = tmp_path / "papers"
    # chunk written at file index i is assigned rank (12 - i): file index 0
    # gets the WORST rank (12), file index 11 gets the BEST rank (1) -- the
    # exact reverse of file order.
    rank_by_artifact = {f"main-paper-{i}": 12 - i for i in range(12)}
    chunks = [
        _main_chunk_with_content(
            f"main-paper-{i}", "Vision main-conference evidence for the shared topic."
        )
        for i in range(12)
    ]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", chunks)

    priority_source_ranks = {"01_topic": rank_by_artifact}

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
        priority_source_ranks=priority_source_ranks,
    )

    composed = result.selected["01_topic"]
    composed_ranks = [rank_by_artifact[str(c["extraction_record_id"])] for c in composed]

    assert composed_ranks == sorted(composed_ranks), (
        f"expected priority chunks in ascending-rank order, got {composed_ranks}"
    )
    best_five_artifacts = {
        artifact_id
        for artifact_id, rank in rank_by_artifact.items()
        if rank <= DEFAULT_MAIN_CONFERENCE_FLOOR
    }
    composed_artifacts = {str(c["extraction_record_id"]) for c in composed}
    assert best_five_artifacts <= composed_artifacts, (
        f"expected the {DEFAULT_MAIN_CONFERENCE_FLOOR} best-ranked papers to all be "
        f"present, missing {best_five_artifacts - composed_artifacts}"
    )


def test_reversed_file_order_yields_identical_selection_with_priority_ids(
    tmp_path: Path,
) -> None:
    """Order-independence extends to the priority-seeded path (Phase
    10.1.1.1): the identical chunk set, written in two physical orders, with
    a priority map present, must yield the identical selected chunk-id SET
    and the identical priority-chunk ORDER. FAILS against the pre-fix
    selector, which has no priority concept and whose composed order is a
    direct function of file-scan order.

    Family sizes are chosen to exactly match what floor-with-backfill
    composition needs (5 priority + 3 keyword-only backfill = exactly
    DEFAULT_MAX_CHUNKS_PER_TOPIC), mirroring
    ``test_reversed_file_order_yields_identical_selection``'s own precedent,
    so no truncation-order ambiguity among equally-scored backfill
    candidates can hide behind this assertion."""

    priority_chunks = [
        _main_chunk_with_content(f"priority-paper-{i}", "Vision priority evidence.")
        for i in range(5)
    ]
    backfill_chunks = [
        _main_chunk_with_content(f"backfill-paper-{i}", "Vision backfill evidence.")
        for i in range(3)
    ]
    priority_source_ranks = {"01_topic": {f"priority-paper-{i}": i + 1 for i in range(5)}}

    def run(chunks: list[dict[str, object]], subdir: str) -> tuple[set[str], list[str]]:
        store_dir = tmp_path / subdir
        _write_chunks_jsonl(store_dir / "chunks.jsonl", chunks)
        result = select_topic_source_chunks(
            store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
            topic_specs=_TOPIC_SPECS,
            compact_chunk=_compact_chunk,
            priority_source_ranks=priority_source_ranks,
        )
        composed = result.selected["01_topic"]
        chunk_ids = {str(c["chunk_id"]) for c in composed}
        priority_order = [
            str(c["extraction_record_id"])
            for c in composed
            if str(c["extraction_record_id"]).startswith("priority-paper-")
        ]
        return chunk_ids, priority_order

    forward_ids, forward_priority_order = run(priority_chunks + backfill_chunks, "forward")
    reversed_ids, reversed_priority_order = run(
        list(reversed(priority_chunks + backfill_chunks)), "reversed"
    )

    assert forward_ids == reversed_ids
    assert forward_priority_order == reversed_priority_order
    assert forward_priority_order, "expected priority chunks to actually be selected"


def test_keyword_backfill_ranked_by_hit_count_not_file_order(tmp_path: Path) -> None:
    """Among non-priority (keyword-backfill) candidates, a chunk hitting
    more distinct topic keywords outranks an earlier-but-weaker match. FAILS
    against the pre-fix selector's boolean ``any(...)`` match, which admits
    both chunks as equally qualified and preserves file order."""

    store_dir = tmp_path / "papers"
    weak_match = _main_chunk_with_content("weak-match", "alpha evidence only.")
    strong_match = _main_chunk_with_content(
        "strong-match", "alpha beta gamma evidence, all three keywords present."
    )
    # weak_match is written FIRST (earlier file position); strong_match is
    # written SECOND, but must still outrank it.
    _write_chunks_jsonl(store_dir / "chunks.jsonl", [weak_match, strong_match])

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_THREE_KEYWORD_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
    )

    composed = result.selected["01_topic"]
    composed_artifacts = [str(c["extraction_record_id"]) for c in composed]
    assert "strong-match" in composed_artifacts and "weak-match" in composed_artifacts
    assert composed_artifacts.index("strong-match") < composed_artifacts.index("weak-match"), (
        f"expected the three-keyword match to be selected ahead of the one-keyword "
        f"match despite arriving later in the file, got order {composed_artifacts}"
    )


def test_priority_paper_selected_even_without_keyword_match(tmp_path: Path) -> None:
    """A priority-named paper is selected even when its text contains none
    of the topic's keywords -- the paid matching signal is authoritative for
    topical relevance and must not be re-litigated by the keyword filter.
    FAILS against the pre-fix selector, whose boolean keyword match admits
    nothing for a chunk with zero keyword hits."""

    store_dir = tmp_path / "papers"
    priority_chunk = _main_chunk_with_content(
        "priority-no-keyword",
        "This abstract discusses an entirely different research area.",
    )
    keyword_chunks = [_main_chunk(i) for i in range(5)]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", [priority_chunk] + keyword_chunks)

    priority_source_ranks = {"01_topic": {"priority-no-keyword": 1}}

    result = select_topic_source_chunks(
        store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
        topic_specs=_TOPIC_SPECS,
        compact_chunk=_compact_chunk,
        priority_source_ranks=priority_source_ranks,
    )

    composed_artifacts = {str(c["extraction_record_id"]) for c in result.selected["01_topic"]}
    assert "priority-no-keyword" in composed_artifacts, (
        f"expected the priority-named paper with zero keyword hits to be selected "
        f"anyway, got {sorted(composed_artifacts)}"
    )


def test_empty_priority_map_reproduces_baseline_selection(tmp_path: Path) -> None:
    """Passing no priority map (``None``, ``{}``, or a per-topic empty dict)
    is a silent no-op that reproduces byte-identical selection -- this is
    what keeps every pre-existing test in this module green without
    modification."""

    store_dir = tmp_path / "papers"
    abundant_chunks = [_main_chunk(i) for i in range(20)] + [
        _workshop_chunk(i) for i in range(20)
    ]
    _write_chunks_jsonl(store_dir / "chunks.jsonl", abundant_chunks)

    def run(priority_source_ranks: object) -> SourcePackSelection:
        return select_topic_source_chunks(
            store_chunk_paths=[("papers", store_dir / "chunks.jsonl")],
            topic_specs=_TOPIC_SPECS,
            compact_chunk=_compact_chunk,
            priority_source_ranks=priority_source_ranks,  # type: ignore[arg-type]
        )

    result_none = run(None)
    result_empty_dict = run({})
    result_empty_topic = run({"01_topic": {}})

    assert result_none.selected == result_empty_dict.selected == result_empty_topic.selected
    assert (
        result_none.family_counts
        == result_empty_dict.family_counts
        == result_empty_topic.family_counts
    )


def test_filler_keywords_are_dropped_and_conference_terms_come_from_the_workspace(tmp_path: Path) -> None:
    from conference_pipeline.analysis.source_pack_selection import conference_keyword_terms, informative_keywords

    kept, dropped = informative_keywords(
        ["icml", "and", "Graph Neural Networks", "the conference", "rl", "diffusion models", "ICML 2026"],
        ignored_terms={"icml", "2026"},
    )
    assert kept == ["graph neural networks", "diffusion models"]
    assert dropped == ["icml", "and", "the conference", "rl", "icml 2026"]

    workspace = tmp_path / "conferences" / "test2026"
    workspace.mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        'conference:\n  slug: "test2026"\n  name: "TEST 2026"\n  full_name: "Test Conference on Testing 2026"\n  year: 2026\n',
        encoding="utf-8",
    )
    terms = conference_keyword_terms(workspace)
    assert {"test2026", "test 2026", "test", "2026", "conference", "testing"} <= terms


def test_identical_selection_groups_are_reported(tmp_path: Path) -> None:
    from conference_pipeline.analysis.source_pack_selection import identical_selection_groups

    groups = identical_selection_groups(
        {
            "01_a": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
            "02_b": [{"chunk_id": "c2"}, {"chunk_id": "c1"}],
            "03_c": [{"chunk_id": "c3"}],
            "04_d": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
            "05_e": [],
        }
    )
    assert groups == [["01_a", "02_b", "04_d"]]
