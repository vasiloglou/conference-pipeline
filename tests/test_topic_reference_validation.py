from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.analysis.topic_reference_validation import (
    build_reference_ledger,
    canonicalize_reference_id,
    run_topic_reference_validation,
    title_aliases,
    validate_reconciliation_references,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True))


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    topic_id = "01_reasoning_post_training"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "topic_id": topic_id,
            "title": "Reasoning and post-training",
            "topic_reconciliation": {
                "status": "succeeded",
                "markdown_path": "analysis/topic_reconciliations/01_reasoning_post_training.md",
            },
            "chunks": [
                {
                    "chunk_id": "speech_transcript:abc123:segment:00001",
                    "extraction_record_id": "speech_transcript:abc123",
                    "source_artifact_path": "media/main.mp3",
                    "talk_title": "Reasoning Cache",
                    "quote": "Reasoning cache reduces repeated thinking.",
                }
            ],
            "llm_full_read_evidence": {
                "selected_items": [
                    {
                        "unit_id": "paper_abstract:2jkAk3EP0v",
                        "input_kind": "paper_abstract",
                        "paper_title": "Latent Thinking Optimization",
                    },
                    {
                        "unit_id": "paper_abstract:1OhgEmix20",
                        "input_kind": "paper_abstract",
                        "paper_title": "LaSeR: Learning to Select Reasoning Paths",
                    },
                ]
            },
            "matched_papers": [],
            "extraction_record_ids": [
                "paper_abstract:2jkAk3EP0v",
                "paper_abstract:1OhgEmix20",
                "speech_transcript:abc123:segment:00001",
            ],
        },
    )
    _write_json(
        workspace
        / "analysis"
        / "exhaustive_summaries"
        / f"{topic_id}__papers.json",
        {
            "topic_id": topic_id,
            "notable_evidence": [
                {
                    "claim": "Latent Thinking Optimization uses latent optimization.",
                    "source_labels": ["Latent Thinking Optimization"],
                    "unit_ids": ["paper_abstract:2jkAk3EP0v"],
                },
                {
                    "claim": "LaSeR selects reasoning paths.",
                    "source_labels": ["LaSeR: Learning to Select Reasoning Paths"],
                    "unit_ids": ["paper_abstract:1OhgEmix20"],
                },
            ],
        },
    )
    _write_json(
        workspace
        / "analysis"
        / "exhaustive_summaries"
        / f"{topic_id}__transcripts.json",
        {
            "topic_id": topic_id,
            "notable_evidence": [
                {
                    "claim": "Reasoning Cache is discussed in a talk.",
                    "source_labels": ["Reasoning Cache keynote"],
                    "unit_ids": ["speech_transcript:abc123:segment:00001"],
                }
            ],
        },
    )
    _write(
        workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md",
        "# Topic Evidence Reconciliation\n\n"
        "Latent Thinking Optimization is cited as `paper_abstract:2jkAk3EP0v`.\n"
        "The talk evidence is `speech_transcript:abc123:segment:00001`.\n",
    )
    return workspace


def _empty_ledger_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    topic_id = "01_reasoning_post_training"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "topic_id": topic_id,
            "title": "Reasoning and post-training",
            "topic_reconciliation": {
                "status": "succeeded",
                "markdown_path": "analysis/topic_reconciliations/01_reasoning_post_training.md",
            },
            "chunks": [],
            "matched_papers": [],
            "extraction_record_ids": [],
        },
    )
    _write(
        workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md",
        "# Topic Evidence Reconciliation\n\nNo canonical citations were available.\n",
    )
    return workspace


def test_cited_full_read_paper_survives_bounded_source_pack(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    topic_id = "01_reasoning_post_training"
    paper_id = "paper_abstract:csf123"
    _write_json(workspace / "analysis" / "analysis_citation_manifest.json", {
        "run_id": "production-run",
        "topics": [{"topic_id": topic_id, "extraction_record_ids": [paper_id]}],
    })
    _write_json(
        workspace / "storage" / "extractions" / "production-run" / "llm-full-read"
        / "by-unit" / "paper_abstract_csf123.json",
        {
            "unit_id": paper_id,
            "source_artifact_path": "storage/papers/csf.pdf",
            "metadata": {"title": "CSF: Black-box Fingerprinting", "authors": ["Junhoo Lee"]},
        },
    )

    ledger = build_reference_ledger(workspace, topic_id)

    assert ledger.entries[paper_id].titles == {"CSF: Black-box Fingerprinting"}
    assert ledger.entries[paper_id].metadata["authors"] == ["Junhoo Lee"]
    assert "storage/papers/csf.pdf" in ledger.entries[paper_id].source_paths
    assert ledger.entries[paper_id].metadata["supplemental_citation_metadata"] is True


def test_supplemental_paper_short_name_does_not_create_false_title_mismatch(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    topic_id = "01_reasoning_post_training"
    paper_id = "paper_abstract:rise123"
    _write_json(workspace / "analysis" / "analysis_citation_manifest.json", {
        "run_id": "production-run",
        "topics": [{"topic_id": topic_id, "extraction_record_ids": [paper_id]}],
    })
    _write_json(
        workspace / "storage" / "extractions" / "production-run" / "llm-full-read"
        / "by-unit" / "paper_abstract_rise123.json",
        {"metadata": {"title": "RISE: Enhancing Image Annotation", "authors": ["A. Author"]}},
    )
    ledger = build_reference_ledger(workspace, topic_id)

    ordinary = validate_reconciliation_references(
        text="Scores rise for `paper_abstract:2jkAk3EP0v`.", ledger=ledger
    )
    full_title = validate_reconciliation_references(
        text="RISE: Enhancing Image Annotation is from `paper_abstract:2jkAk3EP0v`.",
        ledger=ledger,
    )

    assert ordinary.title_mismatch_count == 0
    assert full_title.title_mismatch_count == 1


def test_corpus_paper_title_overrides_foreign_matched_paper_title(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    path = workspace / "analysis" / "narrative_source_packs" / "01_reasoning_post_training.json"
    pack = json.loads(path.read_text(encoding="utf-8"))
    pack["chunks"].append({
        "chunk_id": "paper:abc123:0",
        "extraction_record_id": "paper:abc123",
        "source_artifact_path": "storage/papers/abc123.pdf",
        "paper_title": "Verified Paper on Machine Unlearning",
    })
    pack["matched_papers"] = [
        {"paper_id": "abc123", "title": "Verified Paper on Machine Unlearning"},
        {"paper_id": "abc123", "title": "Unrelated Medical Image Fusion Paper"},
    ]
    _write_json(path, pack)

    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")

    assert ledger.entries["paper_abstract:abc123"].titles == {
        "Verified Paper on Machine Unlearning",
    }


def test_reference_ledger_collects_paper_and_transcript_ids(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")

    assert "paper_abstract:2jkAk3EP0v" in ledger.entries
    assert "paper_abstract:1OhgEmix20" in ledger.entries
    assert "speech_transcript:abc123:segment:00001" in ledger.entries
    assert "Latent Thinking Optimization" in ledger.entries["paper_abstract:2jkAk3EP0v"].titles


def test_reference_validation_catches_id_title_conflation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")
    text = (
        "LaSeR/LTO is cited as `paper_abstract:2jkAk3EP0v`, showing latent "
        "reasoning efficiency."
    )

    result = validate_reconciliation_references(text=text, ledger=ledger)

    assert result.status == "blocked"
    assert result.unknown_reference_count == 0
    assert result.title_mismatch_count == 1
    assert result.title_mismatch_errors[0]["canonical_id"] == "paper_abstract:2jkAk3EP0v"


def test_reference_validation_blocks_unsupported_source_family(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")

    result = validate_reconciliation_references(
        text="The paper text says this directly in `paper_text:BAD`.",
        ledger=ledger,
    )

    assert result.status == "blocked"
    assert result.unknown_reference_ids == ("paper_text:BAD",)
    assert result.unknown_reference_count == 1


def test_reference_validation_reports_unsupported_source_family_on_valid_line(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")

    result = validate_reconciliation_references(
        text=(
            "The validated abstract `paper_abstract:2jkAk3EP0v` agrees with "
            "the extracted full text `paper_text:BAD`."
        ),
        ledger=ledger,
    )

    assert result.status == "blocked"
    assert result.valid_reference_ids == ("paper_abstract:2jkAk3EP0v",)
    assert result.unknown_reference_ids == ("paper_text:BAD",)


def test_reference_validation_blocks_bare_source_ids(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")

    result = validate_reconciliation_references(
        text="Latent Thinking Optimization is cited as `2jkAk3EP0v`.",
        ledger=ledger,
    )

    assert result.status == "blocked"
    assert result.valid_reference_ids == ()
    assert result.citation_count == 0
    assert result.bare_reference_warning_count == 1
    assert result.bare_reference_ids[0]["canonical_id"] == "paper_abstract:2jkAk3EP0v"
    assert {blocker["code"] for blocker in result.blockers} == {"topic_reference_bare_ids"}


def test_reference_validation_blocks_empty_required_ledger(tmp_path: Path) -> None:
    workspace = _empty_ledger_workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")

    result = validate_reconciliation_references(
        text="# Topic Evidence Reconciliation\n\nNo citations.\n",
        ledger=ledger,
    )

    assert result.status == "blocked"
    assert result.ledger_entry_count == 0
    assert result.citation_count == 0
    assert {blocker["code"] for blocker in result.blockers} == {
        "topic_reference_ledger_empty"
    }


def test_reference_validation_writes_receipt_and_enriches_source_pack(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    receipt = run_topic_reference_validation(
        workspace=workspace,
        run_id="test-run",
        topics=["01_reasoning_post_training"],
    )

    assert receipt["status"] == "succeeded"
    assert receipt["checks"]["invalid_reference_count"] == 0
    # v5.1 sub-step lineage: the receipt records what it consumed (07D).
    lineage = receipt["lineage"]
    assert lineage["binding_source"]["substep_id"] == "topic-reference-validation"
    assert [entry["stage_id"] for entry in lineage["upstream_receipts"].values()] == ["topic-reconciliation"]
    assert (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-run"
        / "07E-topic-reference-validation.json"
    ).exists()
    source_pack = json.loads(
        (
            workspace
            / "analysis"
            / "narrative_source_packs"
            / "01_reasoning_post_training.json"
        ).read_text(encoding="utf-8")
    )
    validation = source_pack["topic_reconciliation"]["reference_validation"]
    assert validation["status"] == "succeeded"
    assert validation["ledger_entry_count"] == 4


# ---------------------------------------------------------------------------
# CVPR-style raw corpus-prefixed ids (paper:/workshop_paper:/event_media_caption:)
# regression coverage: these were previously silently dropped from the ledger
# entirely (an empty ledger permanently fails-closed), and populating the
# ledger for the first time surfaced two adjacent defects (a generic-alias
# flood, and cross-conference over-acceptance of malformed prefixes) that this
# suite also pins down.
# ---------------------------------------------------------------------------


def _cvpr_style_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    topic_id = "01_multimodal_foundation_models"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "topic_id": topic_id,
            "title": "Multimodal Foundation Models",
            "chunks": [
                {
                    "chunk_id": "event_media_caption:abc123:0",
                    "extraction_record_id": "event_media_caption:abc123",
                    "source_artifact_path": "storage/download-corpus/event-media/captions/abc123.en.vtt",
                    "quote": "A talk about multimodal grounding.",
                },
                {
                    "chunk_id": "workshop_paper:thecvf.com/CVPR/2026/Workshop/3D4S:PID001:0",
                    "extraction_record_id": "workshop_paper:thecvf.com/CVPR/2026/Workshop/3D4S:PID001",
                    "source_artifact_path": "storage/workshops/papers/3D4S/PID001.pdf",
                    "paper_title": "A Workshop Paper On Grounding",
                    "quote": "Workshop paper abstract text.",
                },
            ],
            "matched_papers": [],
            "extraction_record_ids": [
                "event_media_caption:abc123",
                "workshop_paper:thecvf.com/CVPR/2026/Workshop/3D4S:PID001",
            ],
        },
    )
    _write_json(
        workspace
        / "analysis"
        / "exhaustive_summaries"
        / f"{topic_id}__transcripts.json",
        {
            "topic_id": topic_id,
            "notable_evidence": [
                {
                    "claim": "A talk about multimodal grounding.",
                    # CVPR's exhaustive-summary pipeline attaches this bare
                    # corpus-family name as a placeholder source_label to many
                    # unrelated chunks -- it must never become a disambiguating
                    # alias (that was the alias-flood defect).
                    "source_labels": ["event_media_caption"],
                    "unit_ids": ["event_media_caption:abc123:segment:00000"],
                }
            ],
        },
    )
    _write(
        workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md",
        "# Topic Evidence Reconciliation\n\n"
        "The talk is cited as `speech_transcript:abc123`.\n"
        "The workshop paper is cited as "
        "`paper_abstract:thecvf.com_CVPR_2026_Workshop_3D4S_PID001`.\n",
    )
    return workspace


def test_reference_ledger_canonicalizes_cvpr_raw_prefixes(tmp_path: Path) -> None:
    workspace = _cvpr_style_workspace(tmp_path)

    ledger = build_reference_ledger(workspace, "01_multimodal_foundation_models")

    assert "speech_transcript:abc123" in ledger.entries
    assert (
        "paper_abstract:thecvf.com_CVPR_2026_Workshop_3D4S_PID001" in ledger.entries
    )
    # The chunk-level and (distinct) segment-level ids of the same talk both
    # resolve rather than being silently dropped: the bare chunk-level id from
    # `chunks`/`extraction_record_ids`, plus a genuinely distinct
    # segment-suffixed id from the exhaustive-summary notable_evidence.
    assert "speech_transcript:abc123:segment:00000" in ledger.entries
    assert len(ledger.entries) == 3
    assert ledger.observed_raw_prefixes == frozenset({"event_media_caption", "workshop_paper"})


def test_reference_validation_succeeds_against_cvpr_style_corpus(tmp_path: Path) -> None:
    workspace = _cvpr_style_workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_multimodal_foundation_models")
    text = (workspace / "analysis" / "topic_reconciliations" / "01_multimodal_foundation_models.md").read_text(
        encoding="utf-8"
    )

    result = validate_reconciliation_references(text=text, ledger=ledger, require_citations=True)

    assert result.status == "succeeded"
    assert result.ledger_entry_count == 3
    assert result.unknown_reference_count == 0
    assert result.title_mismatch_count == 0


def test_generic_corpus_family_label_is_not_a_disambiguating_alias() -> None:
    for placeholder in (
        "paper",
        "paper_abstract",
        "workshop_paper",
        "speech_transcript",
        "event_media_caption",
    ):
        assert title_aliases(placeholder) == set(), placeholder
    # A genuine title long enough to qualify is unaffected.
    assert "latent thinking optimization" in title_aliases("Latent Thinking Optimization")


def test_generic_alias_does_not_flood_deterministic_repair(tmp_path: Path) -> None:
    from conference_pipeline.analysis.topic_reference_repair import (
        apply_deterministic_reference_safety_fixes,
    )

    workspace = _cvpr_style_workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_multimodal_foundation_models")
    # A line citing one entry, where the OTHER entry shares the generic
    # "event_media_caption" source_label placeholder -- pre-fix this alias
    # would flag a title_mismatch and get "repaired" by appending the other
    # entry's id; post-fix, the generic label is never an alias at all.
    text = (
        "# Topic Evidence Reconciliation\n\n"
        "The workshop paper is cited as "
        "`paper_abstract:thecvf.com_CVPR_2026_Workshop_3D4S_PID001` "
        "(event_media_caption).\n"
    )

    before = validate_reconciliation_references(text=text, ledger=ledger, require_citations=True)
    repaired_text, changes = apply_deterministic_reference_safety_fixes(text=text, ledger=ledger)

    assert before.title_mismatch_count == 0
    assert changes == 0
    assert repaired_text == text


def test_citation_extraction_scopes_raw_prefix_fallback_to_this_ledger(tmp_path: Path) -> None:
    # CVPR-style ledger: built from raw "workshop_paper:"/"event_media_caption:"
    # ids, so those raw prefixes are in observed_raw_prefixes and a citation
    # using them should still resolve.
    cvpr_workspace = _cvpr_style_workspace(tmp_path)
    cvpr_ledger = build_reference_ledger(cvpr_workspace, "01_multimodal_foundation_models")
    cvpr_result = validate_reconciliation_references(
        text="See `event_media_caption:abc123` for details.",
        ledger=cvpr_ledger,
        require_citations=True,
    )
    assert cvpr_result.valid_reference_ids == ("speech_transcript:abc123",)
    assert cvpr_result.unknown_reference_count == 0

    # ICLR-style ledger: built ONLY from already-canonical paper_abstract:/
    # speech_transcript: ids -- "paper:" is never a legitimate raw prefix here,
    # so a model citing it is genuinely suspect and must NOT be silently
    # normalized into a ledger hit.
    iclr_workspace = _workspace(tmp_path / "iclr_case")
    iclr_ledger = build_reference_ledger(iclr_workspace, "01_reasoning_post_training")
    assert iclr_ledger.observed_raw_prefixes == frozenset({"paper_abstract", "speech_transcript"})
    iclr_result = validate_reconciliation_references(
        text="See `paper:2jkAk3EP0v` for details.",
        ledger=iclr_ledger,
        require_citations=True,
    )
    assert iclr_result.valid_reference_ids == ()
    assert iclr_result.unknown_reference_ids == ("paper:2jkAk3EP0v",)


def test_canonicalize_reference_id_maps_cvpr_prefixes_and_rejects_unknown() -> None:
    assert (
        canonicalize_reference_id("event_media_caption:abc123")
        == "speech_transcript:abc123"
    )
    assert (
        canonicalize_reference_id("event_media_caption:abc123:0")
        == "speech_transcript:abc123"
    )
    assert (
        canonicalize_reference_id("event_media_caption:abc123:segment:00003")
        == "speech_transcript:abc123:segment:00003"
    )
    assert canonicalize_reference_id("paper:xyz") == "paper_abstract:xyz"
    assert (
        canonicalize_reference_id("workshop_paper:host.com/a/b:xyz:0")
        == "paper_abstract:host.com_a_b_xyz"
    )
    # Already-canonical ids pass through unchanged.
    assert canonicalize_reference_id("paper_abstract:xyz") == "paper_abstract:xyz"
    assert (
        canonicalize_reference_id("speech_transcript:xyz:segment:00005")
        == "speech_transcript:xyz:segment:00005"
    )
    # Unrecognized prefixes are rejected, not guessed at.
    assert canonicalize_reference_id("random_prefix:xyz") is None
    assert canonicalize_reference_id("not-a-reference-id") is None


# --- build_reference_ledger: authors forwarded into ReferenceEntry.metadata ---


def _authors_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    topic_id = "01_authors_topic"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "topic_id": topic_id,
            "title": "Authors topic",
            "chunks": [
                {
                    "chunk_id": "paper_abstract:twoauthors",
                    "extraction_record_id": "paper_abstract:twoauthors",
                    "source_artifact_path": "storage/corpus/paper-text/two.txt",
                    "paper_title": "Two Author Paper",
                    "quote": "A single chunk with a two-author fixture.",
                    "authors": ["Ada Lovelace", "Alan Turing"],
                },
                {
                    "chunk_id": "paper_abstract:sharedpaper:segment:00001",
                    "extraction_record_id": "paper_abstract:sharedpaper",
                    "source_artifact_path": "storage/corpus/paper-text/shared.txt",
                    "paper_title": "Shared Paper",
                    "quote": "First chunk of the shared paper.",
                    "authors": ["Ada Lovelace", "Alan Turing"],
                },
                {
                    "chunk_id": "paper_abstract:sharedpaper:segment:00002",
                    "extraction_record_id": "paper_abstract:sharedpaper",
                    "source_artifact_path": "storage/corpus/paper-text/shared.txt",
                    "paper_title": "Shared Paper",
                    "quote": "Second chunk of the shared paper.",
                    "authors": ["Alan Turing", "Grace Hopper"],
                },
                {
                    "chunk_id": "paper_abstract:noauthors",
                    "extraction_record_id": "paper_abstract:noauthors",
                    "source_artifact_path": "storage/corpus/paper-text/noauthors.txt",
                    "paper_title": "No Authors Paper",
                    "quote": "A chunk with no author metadata at all.",
                    "authors": [],
                },
            ],
            "llm_full_read_evidence": {
                "selected_items": [
                    {
                        "unit_id": "paper_abstract:llmitem",
                        "input_kind": "paper_abstract",
                        "paper_title": "LLM-Selected Paper",
                        "authors": ["Katherine Johnson"],
                    }
                ]
            },
            "matched_papers": [],
            "extraction_record_ids": [
                "paper_abstract:twoauthors",
                "paper_abstract:sharedpaper",
                "paper_abstract:noauthors",
                "paper_abstract:llmitem",
            ],
        },
    )
    return workspace


def test_reference_ledger_forwards_chunk_authors_into_metadata(tmp_path: Path) -> None:
    workspace = _authors_workspace(tmp_path)

    ledger = build_reference_ledger(workspace, "01_authors_topic")

    entry = ledger.entries["paper_abstract:twoauthors"]
    assert entry.metadata["authors"] == ["Ada Lovelace", "Alan Turing"]


def test_reference_ledger_merges_overlapping_author_lists_without_duplication(
    tmp_path: Path,
) -> None:
    workspace = _authors_workspace(tmp_path)

    ledger = build_reference_ledger(workspace, "01_authors_topic")

    # Both chunks share extraction_record_id "paper_abstract:sharedpaper";
    # their author lists overlap on "Alan Turing" and must union, not replace,
    # preserving first-seen order with no duplicate name.
    entry = ledger.entries["paper_abstract:sharedpaper"]
    assert entry.metadata["authors"] == [
        "Ada Lovelace",
        "Alan Turing",
        "Grace Hopper",
    ]


def test_reference_ledger_empty_authors_leaves_metadata_without_authors_key(
    tmp_path: Path,
) -> None:
    workspace = _authors_workspace(tmp_path)

    ledger = build_reference_ledger(workspace, "01_authors_topic")

    entry = ledger.entries["paper_abstract:noauthors"]
    assert "authors" not in entry.metadata


def test_reference_ledger_forwards_authors_from_llm_full_read_evidence_selected_items(
    tmp_path: Path,
) -> None:
    workspace = _authors_workspace(tmp_path)

    ledger = build_reference_ledger(workspace, "01_authors_topic")

    entry = ledger.entries["paper_abstract:llmitem"]
    assert entry.metadata["authors"] == ["Katherine Johnson"]


# --- End-to-end regression: raw chunk -> compaction -> ledger metadata ---
#
# This is the Defect 1 regression test required by ROADMAP.md Phase 10.1 done
# gate 1 (test_compaction_and_ledger_preserve_authors_and_title_end_to_end).
# Research Pitfall 3 warns the two fixes (compact_chunk_evidence in
# narrative.py, build_reference_ledger's chunk loop here) must land together
# as one atomic unit -- this test exercises the whole path in one function so
# reverting either half fails it.


def test_compaction_and_ledger_preserve_authors_and_title_end_to_end(
    tmp_path: Path,
) -> None:
    from conference_pipeline.analysis.narrative import compact_chunk_evidence

    topic_id = "01_end_to_end_topic"
    workspace = tmp_path / "workspace"

    raw_chunk = {
        "chunk_id": "paper_abstract:e2epaper",
        "extraction_record_id": "paper_abstract:e2epaper",
        "source_artifact_path": "storage/corpus/paper-text/e2e.txt",
        "verbatim_quote": "word " * 300,  # far over 600 chars once collapsed
        "paper_title": "End To End Regression Paper",
        "authors": ["Ada Lovelace", "Alan Turing"],
    }
    assert len(raw_chunk["verbatim_quote"]) > 1200

    compacted_chunks = compact_chunk_evidence([raw_chunk], limit=8)
    assert len(compacted_chunks[0]["quote"]) == 600

    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "topic_id": topic_id,
            "title": "End to end topic",
            "chunks": compacted_chunks,
            "matched_papers": [],
            "extraction_record_ids": ["paper_abstract:e2epaper"],
        },
    )

    ledger = build_reference_ledger(workspace, topic_id)

    entry = ledger.entries["paper_abstract:e2epaper"]
    assert "End To End Regression Paper" in entry.titles
    assert entry.metadata["authors"] == ["Ada Lovelace", "Alan Turing"]


def test_reference_ledger_input_digest_ignores_source_pack_run_time_stamps(tmp_path: Path) -> None:
    """The ledger (a 07E output covered by the sub-step output digest) must not
    change when an unchanged re-run only restamps the source pack."""

    workspace = _workspace(tmp_path)
    pack_path = workspace / "analysis" / "narrative_source_packs" / "01_reasoning_post_training.json"
    first = build_reference_ledger(workspace, "01_reasoning_post_training").input_digest_sha256
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["topic_reconciliation"]["generated_at"] = "2026-09-18T12:00:00Z"
    pack["generated_at"] = "2026-09-18T12:00:01Z"
    _write_json(pack_path, pack)
    assert build_reference_ledger(workspace, "01_reasoning_post_training").input_digest_sha256 == first
    # Pack enrichment by 07D/07E/repairs is not a ledger input either...
    pack["topic_reconciliation"]["reference_repair"] = {"applied": True}
    pack["topic_reconciliation"]["reference_validation"] = {"status": "succeeded"}
    _write_json(pack_path, pack)
    assert build_reference_ledger(workspace, "01_reasoning_post_training").input_digest_sha256 == first
    pack["llm_full_read_evidence"]["receipt_sha256"] = "f" * 64  # wrapper digest, not read
    _write_json(pack_path, pack)
    assert build_reference_ledger(workspace, "01_reasoning_post_training").input_digest_sha256 == first
    # ...but the evidence the ledger is built from is.
    pack["chunks"][0]["quote"] = "A different quote."
    _write_json(pack_path, pack)
    assert build_reference_ledger(workspace, "01_reasoning_post_training").input_digest_sha256 != first


def test_scoped_validation_is_left_unbound_and_repeat_runs_keep_their_digest(tmp_path: Path) -> None:
    """A topic-subset 07E run is not bound as current (other topics were not
    re-validated); two consecutive full runs on unchanged evidence produce the
    same ledger and output digest (07E's own source-pack enrichment is not a
    ledger input)."""

    import shutil

    workspace = _workspace(tmp_path)
    packs = workspace / "analysis" / "narrative_source_packs"
    other = packs / "02_other_topic.json"
    shutil.copy(packs / "01_reasoning_post_training.json", other)
    pack = json.loads(other.read_text(encoding="utf-8"))
    pack["topic_id"] = "02_other_topic"
    _write_json(other, pack)
    source = workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md"
    shutil.copy(source, source.with_name("02_other_topic.md"))

    scoped = run_topic_reference_validation(workspace=workspace, run_id="test-run", topics=["01_reasoning_post_training"])
    assert "lineage" not in scoped and scoped["lineage_unbound_reason"] == "partial_scope"

    first = run_topic_reference_validation(workspace=workspace, run_id="test-run")
    second = run_topic_reference_validation(workspace=workspace, run_id="test-run")
    assert first["status"] == second["status"] == "succeeded"
    assert first["lineage"]["binding_source"]["substep_id"] == "topic-reference-validation"
    assert second["substep_outputs"]["digest"] == first["substep_outputs"]["digest"]
    first_ledger = build_reference_ledger(workspace, "01_reasoning_post_training").input_digest_sha256
    run_topic_reference_validation(workspace=workspace, run_id="test-run")
    assert build_reference_ledger(workspace, "01_reasoning_post_training").input_digest_sha256 == first_ledger


def test_kind_noun_source_labels_are_not_disambiguating_aliases() -> None:
    """Live CVPR 2026 topic 08: whole families of entries carried the labels
    "workshop"/"proceedings"/"talk", so every line that cited one entry and
    used the word "workshop" was a title mismatch."""

    for label in ("workshop", "Workshop", "talk", "proceedings", "keynote", "tutorial"):
        assert title_aliases(label) == set(), label
    assert "seele" in title_aliases("Seele: A Unified Acceleration Framework")


def _shared_name_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conf"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / "01_topic.json",
        {
            "topic_id": "01_topic",
            "title": "Topic One",
            "chunks": [
                {
                    "chunk_id": "event_media_caption:seelevid:0",
                    "extraction_record_id": "event_media_caption:seelevid",
                    "talk_title": "Seele: Real-Time Gaussian Splatting On Mobile",
                    "quote": "The talk presenting Seele.",
                }
            ],
            "llm_full_read_evidence": {
                "selected_items": [
                    {
                        "unit_id": "paper_abstract:ff9db885",
                        "input_kind": "paper_abstract",
                        "paper_title": "Seele: A Unified Acceleration Framework",
                    }
                ]
            },
            "matched_papers": [
                {"paper_id": "14", "title": "Villa-x: latent action modeling"},
                {"paper_id": "abc123def456", "title": "A Properly Bound Paper"},
            ],
            "extraction_record_ids": ["event_media_caption:seelevid"],
        },
    )
    return workspace


def test_alias_shared_by_a_cited_entry_is_not_foreign(tmp_path: Path) -> None:
    """A paper and the talk presenting it share a name; a line citing either
    may use that name. Pre-fix the shared alias counted as foreign and the
    deterministic strip deleted the word from a quotation."""

    ledger = build_reference_ledger(_shared_name_workspace(tmp_path), "01_topic")
    text = (
        "# Topic Evidence Reconciliation\n\n"
        '* "Seele achieves real-time splatting" (`speech_transcript:seelevid`).\n'
    )
    result = validate_reconciliation_references(text=text, ledger=ledger, require_citations=True)
    assert result.title_mismatch_count == 0
    assert result.unknown_reference_count == 0


def test_ordinal_matched_paper_ids_are_left_out_of_the_ledger(tmp_path: Path) -> None:
    """Paper matching leaves an ordinal when it cannot bind a paper to the
    corpus. As a ledger entry its bare id "14" matched every "14" in prose
    as a bare-id mention (live CVPR 2026 topic 05 blocked on "~7 FPS")."""

    ledger = build_reference_ledger(_shared_name_workspace(tmp_path), "01_topic")
    assert "paper_abstract:14" not in ledger.entries
    assert "paper_abstract:abc123def456" in ledger.entries
    text = (
        "# Topic Evidence Reconciliation\n\n"
        "* Requires ~14 FPS on average (`paper_abstract:abc123def456`).\n"
    )
    result = validate_reconciliation_references(text=text, ledger=ledger, require_citations=True)
    assert result.bare_reference_warning_count == 0
