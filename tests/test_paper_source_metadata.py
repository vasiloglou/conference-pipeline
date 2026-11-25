"""Unit tests for the shared paper title/author metadata resolution module.

Covers: source-family classification, the O(1) join index, the two
authoritative resolvers (source-discovery join, workshop receipt), record
enrichment, author-name normalization, the deterministic PDF-header fallback
parser (split-before-strip ordering fix and its observed failure modes), and
the reproducible seeded measurement command.
"""

from __future__ import annotations

import json

from conference_pipeline.indexing.paper_source_metadata import (
    PROVENANCE_HEADER_PARSE,
    PROVENANCE_SOURCE_DISCOVERY_JOIN,
    PROVENANCE_UNRESOLVED,
    PROVENANCE_WORKSHOP_RECEIPT,
    SOURCE_FAMILY_CLASS_ARTIFACT,
    SOURCE_FAMILY_CLASS_MAIN_PAPER,
    SOURCE_FAMILY_CLASS_WORKSHOP_PAPER,
    SourceMetadataResolution,
    SOURCE_FAMILY_CLASSES,
    build_program_item_index,
    classify_record_family,
    enrich_paper_record,
    enrich_paper_records,
    measure_headers,
    normalize_author_names,
    parse_paper_header_metadata,
    resolve_main_paper_metadata,
    resolve_workshop_paper_metadata,
)


def _sample_join_fixture() -> tuple[dict, dict]:
    download_corpus_ledger = {
        "program_item_results": [
            {"request_key": "rk-1", "program_item_id": "pi-1"},
            {"request_key": "rk-2", "program_item_id": "pi-2"},
            {"request_key": "rk-3", "program_item_id": "pi-3"},
        ]
    }
    source_discovery_inventory = {
        "program_items": [
            {"program_item_id": "pi-1", "field_evidence": {"title": {"source_record_id": "sr-1"}}},
            {"program_item_id": "pi-2", "field_evidence": {"title": {"source_record_id": "sr-2"}}},
            {"program_item_id": "pi-3", "field_evidence": {"title": {"source_record_id": "sr-3"}}},
        ],
        "source_records": [
            {"source_record_id": "sr-1", "title": "Paper One", "authors": ["Ann Lee", "Bob Kim"]},
            {"source_record_id": "sr-2", "title": "Paper Two"},
            {"source_record_id": "sr-3", "title": "Paper Three", "authors": ["Carl Doe"]},
        ],
    }
    return download_corpus_ledger, source_discovery_inventory


# ---------------------------------------------------------------------------
# Task 1: family classification
# ---------------------------------------------------------------------------


def test_classify_record_family_main_paper() -> None:
    assert classify_record_family({"source_family_id": "paper_text"}) == SOURCE_FAMILY_CLASS_MAIN_PAPER


def test_classify_record_family_workshop_paper() -> None:
    assert (
        classify_record_family({"source_family_id": "workshop_paper_text"})
        == SOURCE_FAMILY_CLASS_WORKSHOP_PAPER
    )


def test_classify_record_family_artifact_family_and_unknown() -> None:
    assert classify_record_family({"source_family_id": "speech_transcript"}) == SOURCE_FAMILY_CLASS_ARTIFACT
    assert classify_record_family({"source_family_id": "totally_unknown"}) == SOURCE_FAMILY_CLASS_ARTIFACT
    assert classify_record_family({}) == SOURCE_FAMILY_CLASS_ARTIFACT


# ---------------------------------------------------------------------------
# Task 1: O(1) join index
# ---------------------------------------------------------------------------


def test_build_program_item_index_o1_lookup_and_idempotent() -> None:
    ledger, inventory = _sample_join_fixture()

    index_a = build_program_item_index(download_corpus_ledger=ledger, source_discovery_inventory=inventory)
    index_b = build_program_item_index(download_corpus_ledger=ledger, source_discovery_inventory=inventory)

    assert isinstance(index_a.program_item_id_by_request_key, dict)
    assert index_a.program_item_id_by_request_key == index_b.program_item_id_by_request_key
    assert index_a.program_items_by_id == index_b.program_items_by_id
    assert index_a.source_records_by_id == index_b.source_records_by_id
    assert index_a.program_item_id_by_request_key["rk-1"] == "pi-1"


def test_build_program_item_index_tolerates_malformed_rows() -> None:
    ledger = {
        "program_item_results": [
            "not-a-mapping",
            {"request_key": None, "program_item_id": "pi-1"},
            {"program_item_id": "pi-missing-request-key"},
            {"request_key": "rk-1", "program_item_id": "pi-1"},
        ]
    }
    inventory = {
        "program_items": ["not-a-mapping", {"program_item_id": "pi-1", "field_evidence": {}}],
        "source_records": [{"source_record_id": "sr-1", "title": "T"}],
    }

    index = build_program_item_index(download_corpus_ledger=ledger, source_discovery_inventory=inventory)

    assert index.program_item_id_by_request_key == {"rk-1": "pi-1"}
    assert "pi-1" in index.program_items_by_id


# ---------------------------------------------------------------------------
# Task 1: resolve_main_paper_metadata
# ---------------------------------------------------------------------------


def test_resolve_main_paper_metadata_resolves_title_and_authors() -> None:
    ledger, inventory = _sample_join_fixture()
    index = build_program_item_index(download_corpus_ledger=ledger, source_discovery_inventory=inventory)

    resolution = resolve_main_paper_metadata("rk-1", index)

    assert resolution.title == "Paper One"
    assert resolution.authors == ("Ann Lee", "Bob Kim")
    assert resolution.title_provenance == PROVENANCE_SOURCE_DISCOVERY_JOIN
    assert resolution.authors_provenance == PROVENANCE_SOURCE_DISCOVERY_JOIN


def test_resolve_main_paper_metadata_missing_authors_key() -> None:
    ledger, inventory = _sample_join_fixture()
    index = build_program_item_index(download_corpus_ledger=ledger, source_discovery_inventory=inventory)

    resolution = resolve_main_paper_metadata("rk-2", index)

    assert resolution.title == "Paper Two"
    assert resolution.title_provenance == PROVENANCE_SOURCE_DISCOVERY_JOIN
    assert resolution.authors == ()
    assert resolution.authors_provenance == PROVENANCE_UNRESOLVED


def test_resolve_main_paper_metadata_missing_request_key_no_exception() -> None:
    ledger, inventory = _sample_join_fixture()
    index = build_program_item_index(download_corpus_ledger=ledger, source_discovery_inventory=inventory)

    resolution = resolve_main_paper_metadata("rk-does-not-exist", index)

    assert resolution == SourceMetadataResolution.unresolved()


def test_resolve_main_paper_metadata_prefers_symmetric_authors_evidence() -> None:
    ledger = {"program_item_results": [{"request_key": "rk-1", "program_item_id": "pi-1"}]}
    inventory = {
        "program_items": [
            {
                "program_item_id": "pi-1",
                "field_evidence": {
                    "title": {"source_record_id": "sr-title"},
                    "authors": {"source_record_id": "sr-authors"},
                },
            }
        ],
        "source_records": [
            {"source_record_id": "sr-title", "title": "Paper One", "authors": ["Wrong Person"]},
            {"source_record_id": "sr-authors", "authors": ["Ann Lee", "Bob Kim"]},
        ],
    }
    index = build_program_item_index(download_corpus_ledger=ledger, source_discovery_inventory=inventory)

    resolution = resolve_main_paper_metadata("rk-1", index)

    assert resolution.title == "Paper One"
    assert resolution.authors == ("Ann Lee", "Bob Kim")


# ---------------------------------------------------------------------------
# Task 1: resolve_workshop_paper_metadata
# ---------------------------------------------------------------------------


def test_resolve_workshop_paper_metadata_with_title_and_authors() -> None:
    resolution = resolve_workshop_paper_metadata({"title": "T", "authors": ["A", "B"]})

    assert resolution.title == "T"
    assert resolution.authors == ("A", "B")
    assert resolution.title_provenance == PROVENANCE_WORKSHOP_RECEIPT
    assert resolution.authors_provenance == PROVENANCE_WORKSHOP_RECEIPT


def test_resolve_workshop_paper_metadata_title_only() -> None:
    resolution = resolve_workshop_paper_metadata({"title": "T"})

    assert resolution.title == "T"
    assert resolution.title_provenance == PROVENANCE_WORKSHOP_RECEIPT
    assert resolution.authors == ()
    assert resolution.authors_provenance == PROVENANCE_UNRESOLVED


# ---------------------------------------------------------------------------
# Task 1: enrich_paper_record
# ---------------------------------------------------------------------------


def test_enrich_paper_record_writes_fields_and_preserves_existing_title() -> None:
    resolution_full = SourceMetadataResolution(
        title="New Title",
        authors=("Ann Lee", "Bob Kim"),
        title_provenance=PROVENANCE_SOURCE_DISCOVERY_JOIN,
        authors_provenance=PROVENANCE_SOURCE_DISCOVERY_JOIN,
    )
    record: dict = {}
    enrich_paper_record(record, resolution_full)

    assert record["paper_title"] == "New Title"
    assert record["authors"] == ["Ann Lee", "Bob Kim"]
    assert record["source_metadata_provenance"] == {
        "title": PROVENANCE_SOURCE_DISCOVERY_JOIN,
        "authors": PROVENANCE_SOURCE_DISCOVERY_JOIN,
    }

    existing_record = {"paper_title": "Existing Title"}
    enrich_paper_record(existing_record, SourceMetadataResolution.unresolved())

    assert existing_record["paper_title"] == "Existing Title"
    assert "authors" not in existing_record
    assert existing_record["source_metadata_provenance"] == {
        "title": PROVENANCE_UNRESOLVED,
        "authors": PROVENANCE_UNRESOLVED,
    }


# ---------------------------------------------------------------------------
# Plan 10.1-05 Task 1: enrich_paper_records (shared batch entry point)
# ---------------------------------------------------------------------------


def _mixed_fixture() -> tuple[list[dict], "object"]:
    download_corpus_ledger, source_discovery_inventory = _sample_join_fixture()
    index = build_program_item_index(
        download_corpus_ledger=download_corpus_ledger,
        source_discovery_inventory=source_discovery_inventory,
    )
    records = [
        {"artifact_id": "paper:rk-1", "source_family_id": "paper_text"},
        {"artifact_id": "workshop_paper:venue-x:note-1", "source_family_id": "workshop_paper_text"},
        {"artifact_id": "speech:vid-1", "source_family_id": "speech_transcript"},
    ]
    return records, index


def test_enrich_paper_records_coverage_has_exactly_three_family_keys() -> None:
    records, index = _mixed_fixture()

    _records, coverage = enrich_paper_records(records, index=index)

    assert set(coverage) == set(SOURCE_FAMILY_CLASSES)
    assert set(coverage) == {"papers_main", "papers_workshop", "artifacts"}


def test_enrich_paper_records_main_and_workshop_both_resolve_authors_in_one_call() -> None:
    records, index = _mixed_fixture()
    workshop_receipts = {
        "workshop_paper:venue-x:note-1": {
            "title": "Workshop Fixture Paper",
            "authors": ["Dana Fox", "Evan Ray"],
        }
    }

    enriched, coverage = enrich_paper_records(
        records, index=index, workshop_receipts=workshop_receipts
    )

    main_record = next(r for r in enriched if r["artifact_id"] == "paper:rk-1")
    workshop_record = next(
        r for r in enriched if r["artifact_id"] == "workshop_paper:venue-x:note-1"
    )
    assert main_record["authors"] == ["Ann Lee", "Bob Kim"]
    assert main_record["source_metadata_provenance"]["authors"] == PROVENANCE_SOURCE_DISCOVERY_JOIN
    assert workshop_record["authors"] == ["Dana Fox", "Evan Ray"]
    assert workshop_record["source_metadata_provenance"]["authors"] == PROVENANCE_WORKSHOP_RECEIPT
    assert coverage["papers_main"]["authors_resolved"] == 1
    assert coverage["papers_workshop"]["authors_resolved"] == 1


def test_enrich_paper_records_artifact_family_passed_through_and_counted_honestly() -> None:
    records = [
        {"artifact_id": "speech:vid-1", "source_family_id": "speech_transcript"},
        {
            "artifact_id": "speech:vid-2",
            "source_family_id": "speech_transcript",
            "paper_title": "Pre-existing artifact title",
        },
    ]

    enriched, coverage = enrich_paper_records(records)

    assert enriched[0] == {"artifact_id": "speech:vid-1", "source_family_id": "speech_transcript"}
    assert coverage["artifacts"]["total"] == 2
    assert coverage["artifacts"]["title_resolved"] == 1
    assert coverage["artifacts"]["authors_resolved"] == 0


def test_enrich_paper_records_unresolved_record_gains_no_fabricated_values() -> None:
    records = [{"artifact_id": "paper:not-in-index", "source_family_id": "paper_text"}]

    enriched, coverage = enrich_paper_records(records, index=None)

    record = enriched[0]
    assert "paper_title" not in record
    assert "authors" not in record
    assert record["source_metadata_provenance"] == {
        "title": PROVENANCE_UNRESOLVED,
        "authors": PROVENANCE_UNRESOLVED,
    }
    assert coverage["papers_main"]["title_resolved"] == 0
    assert coverage["papers_main"]["authors_resolved"] == 0
    assert coverage["papers_main"]["provenance"]["title"][PROVENANCE_UNRESOLVED] == 1


def test_enrich_paper_records_no_blended_aggregate_key() -> None:
    records, index = _mixed_fixture()

    _records, coverage = enrich_paper_records(records, index=index)

    assert "total" not in coverage
    assert "aggregate" not in coverage
    assert all(isinstance(v, dict) for v in coverage.values())


def test_enrich_paper_records_no_filesystem_access_when_reader_is_none(monkeypatch) -> None:
    records, index = _mixed_fixture()

    def _raising_open(*_args, **_kwargs):
        raise AssertionError("enrich_paper_records must not touch the filesystem")

    monkeypatch.setattr("builtins.open", _raising_open)

    enrich_paper_records(records, index=index)


def test_enrich_paper_records_idempotent_across_two_calls() -> None:
    records, index = _mixed_fixture()
    workshop_receipts = {
        "workshop_paper:venue-x:note-1": {
            "title": "Workshop Fixture Paper",
            "authors": ["Dana Fox"],
        }
    }

    first_records, first_coverage = enrich_paper_records(
        records, index=index, workshop_receipts=workshop_receipts
    )
    second_records, second_coverage = enrich_paper_records(
        first_records, index=index, workshop_receipts=workshop_receipts
    )

    assert first_records == second_records
    assert first_coverage == second_coverage


def test_enrich_paper_records_header_fallback_only_fills_missing_components() -> None:
    records = [{"artifact_id": "paper:not-in-index", "source_family_id": "paper_text"}]

    def _reader(_record: dict) -> str:
        return "Header Parsed Title\nAnn Author1\nAbstract\n"

    enriched, coverage = enrich_paper_records(records, header_text_reader=_reader)

    record = enriched[0]
    assert record["source_metadata_provenance"]["title"] == PROVENANCE_HEADER_PARSE
    assert coverage["papers_main"]["title_resolved"] == 1


# ---------------------------------------------------------------------------
# Task 1: normalize_author_names
# ---------------------------------------------------------------------------


def test_normalize_author_names_dedup_strip_order() -> None:
    result = normalize_author_names([" Ann  Lee ", "Bob Kim", "", None, "Ann Lee", "Bob Kim"])

    assert result == ("Ann Lee", "Bob Kim")


# ---------------------------------------------------------------------------
# Task 2: parse_paper_header_metadata (deterministic PDF-header fallback)
# ---------------------------------------------------------------------------


def test_header_ordering_fix_glued_superscripts_no_comma() -> None:
    header = "Deep Learning For Something\nAobo Li1 Jinjian Wu2\nUniversity Somewhere\nAbstract\nBody text."

    resolution = parse_paper_header_metadata(header)

    assert resolution.authors == ("Aobo Li", "Jinjian Wu")
    assert resolution.authors_provenance == PROVENANCE_HEADER_PARSE


def test_header_comma_separated_markers_with_asterisk() -> None:
    header = "A Great Paper Title\nXiaofan Que1,2*, Wei Chen3\nSome Affiliation\nAbstract\nBody."

    resolution = parse_paper_header_metadata(header)

    assert resolution.authors == ("Xiaofan Que", "Wei Chen")


def test_header_two_physical_line_title_joins_with_single_space() -> None:
    header = "Robust Visual\nRepresentation Learning\nJane Doe1 John Roe2\nSome University\nAbstract\nBody."

    resolution = parse_paper_header_metadata(header)

    assert resolution.title == "Robust Visual Representation Learning"


def test_header_ligature_and_spacing_artifact_normalization() -> None:
    header = (
        "A Study of Qi Y u Networks for classiﬁcation\n"
        "Ann Lee1 Bob Kim2\n"
        "Some Lab\n"
        "Abstract\n"
        "Body."
    )

    resolution = parse_paper_header_metadata(header)

    assert resolution.title == "A Study of Qi Yu Networks for classification"
    assert resolution.authors == ("Ann Lee", "Bob Kim")


def test_header_affiliation_line_without_markers_not_treated_as_authors() -> None:
    header = "Some Interesting Title\nCambridge University Research Lab\nCambridge MA USA\nAbstract\nBody."

    resolution = parse_paper_header_metadata(header)

    assert resolution.authors == ()
    assert resolution.authors_provenance == PROVENANCE_UNRESOLVED


def test_header_missing_abstract_marker_still_returns_title() -> None:
    lines = ["A Title With No Abstract Marker"] + [f"Filler line number {i}" for i in range(20)]
    header = "\n".join(lines)

    resolution = parse_paper_header_metadata(header, max_header_lines=16)

    assert resolution.title is not None
    assert resolution.title.startswith("A Title With No Abstract Marker")


def test_header_bounded_scan_window_handles_5mb_input_quickly() -> None:
    import time

    huge_text = "Padding Title Line\nAnn Lee1 Bob Kim2\nLab\n" + ("word " * 1_000_000)
    assert len(huge_text) > 5_000_000

    start = time.perf_counter()
    resolution = parse_paper_header_metadata(huge_text)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert resolution.authors == ("Ann Lee", "Bob Kim")


def test_header_provenance_labels_are_pdf_header_parse_when_resolved() -> None:
    header = "A Great Paper\nAobo Li1 Jinjian Wu2\nLab\nAbstract\nBody."

    resolution = parse_paper_header_metadata(header)

    assert resolution.title_provenance == PROVENANCE_HEADER_PARSE
    assert resolution.authors_provenance == PROVENANCE_HEADER_PARSE


def test_header_no_input_returns_fully_unresolved_without_raising() -> None:
    resolution = parse_paper_header_metadata("")

    assert resolution == SourceMetadataResolution.unresolved()


# ---------------------------------------------------------------------------
# Task 3: measure_headers (reproducible seeded measurement command)
# ---------------------------------------------------------------------------


def test_measure_headers_counts_three_crafted_files(tmp_path) -> None:
    (tmp_path / "file_a.txt").write_text(
        "A Wonderful Paper\nAobo Li1 Jinjian Wu2\nSome Lab\nAbstract\nBody text.\n"
    )
    (tmp_path / "file_b.txt").write_text(
        "Another Great Study\nCambridge University Lab\nCambridge MA USA\nAbstract\nBody text.\n"
    )
    (tmp_path / "file_c.txt").write_text(
        "Solo Author Paper\nJane Doe1\nSolo Lab\nAbstract\nBody text.\n"
    )

    result = measure_headers(str(tmp_path), sample=300, seed=42)

    assert result["sample_size"] == 3
    assert result["skipped"] == 0
    assert result["title_resolved"] == 3
    assert result["authors_any_resolved"] == 2
    assert result["authors_two_or_more_resolved"] == 1
    assert result["abstract_marker_found"] == 3
    assert result["fully_clean"] == 1


def test_measure_headers_same_seed_is_byte_identical(tmp_path) -> None:
    (tmp_path / "file_a.txt").write_text("Title One\nAnn Lee1 Bob Kim2\nLab\nAbstract\nBody.\n")
    (tmp_path / "file_b.txt").write_text("Title Two\nCarl Doe1 Dana Roe2\nLab\nAbstract\nBody.\n")

    first = json.dumps(measure_headers(str(tmp_path), sample=10, seed=7), sort_keys=True)
    second = json.dumps(measure_headers(str(tmp_path), sample=10, seed=7), sort_keys=True)

    assert first == second


def test_measure_headers_rejects_symlinked_root(tmp_path) -> None:
    real_target = tmp_path / "outside"
    real_target.mkdir()
    (real_target / "secret.txt").write_text(
        "Secret Title\nAobo Li1 Jinjian Wu2\nLab\nAbstract\nBody.\n"
    )
    root_link = tmp_path / "root_link"
    root_link.symlink_to(real_target)

    result = measure_headers(str(root_link))

    assert "error" in result
    assert len(result["error"]) > 0
    assert "sample_size" not in result
