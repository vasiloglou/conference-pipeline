from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.analysis.topic_reference_repair import (
    MAX_MISSING_REFERENCE_IDS_PER_LINE,
    apply_deterministic_reference_safety_fixes,
    collect_repair_input,
    run_topic_reference_repair,
)
from conference_pipeline.analysis.topic_reference_validation import (
    build_reference_ledger,
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
                    "source_artifact_path": "media/main.mp3",
                    "talk_title": "Reasoning Cache",
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
        workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.json",
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
    _write(
        workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md",
        "# Topic Evidence Reconciliation\n\n"
        "LaSeR is cited as `paper_abstract:2jkAk3EP0v`.\n"
        "A bare segment appears as `abc123:segment:00001`.\n",
    )
    return workspace


class _FakeResearcher:
    model = "fake-qwen"
    provider_receipt_paths: list[str] = []

    async def run(self, query: str, *, context_path: Path | None = None) -> str:
        assert "Validator Errors To Repair" in query
        assert "LaSeR" in query
        return (
            "# Topic Evidence Reconciliation\n\n"
            "LaSeR is cited as `paper_abstract:1OhgEmix20`.\n"
            "The talk evidence is `speech_transcript:abc123:segment:00001`.\n"
        )

    def usage(self) -> dict[str, object]:
        return {"request_count": 1, "estimated_cost_usd": 0.0}


def test_collect_repair_input_includes_validation_errors_and_ledger(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    repair_input = collect_repair_input(workspace, topic_id="01_reasoning_post_training")

    assert repair_input.initial_validation.status == "blocked"
    assert "paper_abstract:1OhgEmix20" in repair_input.prompt
    assert "Validator Errors To Repair" in repair_input.prompt


def test_reference_repair_applies_only_after_revalidation_passes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    receipt = run_topic_reference_repair(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        topics=["01_reasoning_post_training"],
        apply_repaired=True,
        researcher=_FakeResearcher(),
    )

    assert receipt["status"] == "succeeded"
    assert receipt["checks"]["initial_invalid_reference_count"] == 2
    assert receipt["checks"]["repaired_invalid_reference_count"] == 0
    repaired_text = (
        workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md"
    ).read_text(encoding="utf-8")
    result = validate_reconciliation_references(
        text=repaired_text,
        ledger=build_reference_ledger(workspace, "01_reasoning_post_training"),
    )
    assert result.status == "succeeded"
    source_pack = json.loads(
        (
            workspace
            / "analysis"
            / "narrative_source_packs"
            / "01_reasoning_post_training.json"
        ).read_text(encoding="utf-8")
    )
    assert source_pack["topic_reconciliation"]["reference_repair"]["status"] == "succeeded"


def test_deterministic_safety_fixes_expand_segment_shorthand_without_duplication(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")
    text = (
        "`speech_transcript:abc123:segment:00000` / `segment:00001` "
        "(Reasoning Cache): adjacent segment evidence.\n"
    )

    repaired, change_count = apply_deterministic_reference_safety_fixes(
        text=text,
        ledger=ledger,
    )

    assert change_count >= 1
    assert "abc123:speech_transcript:abc123" not in repaired
    assert "`speech_transcript:abc123:segment:00001`" in repaired


def test_deterministic_safety_fixes_adds_missing_shared_alias_ids(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    topic_id = "01_reasoning_post_training"
    source_pack_path = (
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    )
    source_pack = json.loads(source_pack_path.read_text(encoding="utf-8"))
    source_pack["extraction_record_ids"].extend(
        ["paper_abstract:spiceA", "paper_abstract:spiceB"]
    )
    _write_json(source_pack_path, source_pack)
    papers_path = (
        workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.json"
    )
    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    papers["notable_evidence"].extend(
        [
            {
                "claim": "SPICE has one relevant abstract.",
                "source_labels": ["SPICE"],
                "unit_ids": ["paper_abstract:spiceA"],
            },
            {
                "claim": "SPICE has another relevant abstract.",
                "source_labels": ["SPICE"],
                "unit_ids": ["paper_abstract:spiceB"],
            },
        ]
    )
    _write_json(papers_path, papers)
    ledger = build_reference_ledger(workspace, topic_id)
    # Since d5d3988edf an alias is foreign only when NONE of the ids cited on
    # the line owns it (a paper and the talk presenting it share a name), so a
    # line citing one SPICE entry may say "SPICE" without citing the other.
    owner_text = "SPICE is a key example. (`paper_abstract:spiceA`)\n"
    assert validate_reconciliation_references(text=owner_text, ledger=ledger).status == "succeeded"
    # A line naming SPICE while citing an unrelated paper is still a mismatch,
    # and the deterministic fix appends every id that owns the shared alias.
    text = "SPICE is a key example. (`paper_abstract:2jkAk3EP0v`)\n"

    initial = validate_reconciliation_references(text=text, ledger=ledger)
    repaired, change_count = apply_deterministic_reference_safety_fixes(
        text=text,
        ledger=ledger,
    )
    result = validate_reconciliation_references(text=repaired, ledger=ledger)

    assert initial.status == "blocked"
    assert change_count >= 1
    assert "`paper_abstract:spiceA`" in repaired
    assert "`paper_abstract:spiceB`" in repaired
    assert result.status == "succeeded"


def test_deterministic_safety_fixes_refuses_to_flood_a_line_with_many_foreign_ids(
    tmp_path: Path,
) -> None:
    """Regression: a generic/non-discriminating alias (e.g. a placeholder
    source_label shared by many chunks) resolving to more than a handful of
    foreign ids is evidence the alias itself is non-discriminating, not that
    one line is genuinely missing that many citations. The deterministic
    "fix" must leave such a line (and its title_mismatch blocker) untouched
    rather than appending every foreign id -- confirmed against the live
    CVPR2026 corpus to otherwise flood a line with 20-100 unrelated ids and
    falsely flip a topic's validation status to "succeeded"."""

    workspace = _workspace(tmp_path)
    topic_id = "01_reasoning_post_training"
    source_pack_path = (
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    )
    source_pack = json.loads(source_pack_path.read_text(encoding="utf-8"))
    # The line below cites an unrelated paper, so every one of these ids is
    # "missing"; the count must exceed the cap, not merely equal it.
    extra_ids = [f"paper_abstract:shared{i}" for i in range(MAX_MISSING_REFERENCE_IDS_PER_LINE + 2)]
    source_pack["extraction_record_ids"].extend(extra_ids)
    _write_json(source_pack_path, source_pack)
    papers_path = (
        workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.json"
    )
    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    papers["notable_evidence"].extend(
        [
            {
                "claim": f"Shared claim {i}.",
                "source_labels": ["Shared Non Discriminating Label"],
                "unit_ids": [ref_id],
            }
            for i, ref_id in enumerate(extra_ids)
        ]
    )
    _write_json(papers_path, papers)
    ledger = build_reference_ledger(workspace, topic_id)
    # The line must literally mention the shared alias text for
    # validate_reconciliation_references to detect the foreign-alias mismatch
    # in the first place (matching how the real CVPR bug manifested: the
    # generic source_label text appeared verbatim in the model's own
    # reconciliation prose).
    # It cites an id that does NOT own the alias: since d5d3988edf a line
    # citing any owner of a shared alias is not a mismatch at all.
    text = "Shared Non Discriminating Label example. (`paper_abstract:2jkAk3EP0v`)\n"

    initial = validate_reconciliation_references(text=text, ledger=ledger)
    repaired, change_count = apply_deterministic_reference_safety_fixes(
        text=text,
        ledger=ledger,
    )

    assert initial.status == "blocked"
    assert initial.title_mismatch_count >= 1
    # Refused to flood: text is unchanged, none of the foreign ids appended.
    assert repaired == text
    assert change_count == 0


def test_deterministic_safety_fixes_canonicalizes_bare_source_details(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    ledger = build_reference_ledger(workspace, "01_reasoning_post_training")
    text = (
        "- Presentation ID: 2jkAk3EP0v\n"
        "- Timestamp: abc123:segment:00001\n"
    )

    repaired, change_count = apply_deterministic_reference_safety_fixes(
        text=text,
        ledger=ledger,
    )
    result = validate_reconciliation_references(text=repaired, ledger=ledger)

    assert change_count >= 2
    assert "`paper_abstract:2jkAk3EP0v`" in repaired
    assert "speech_transcript:abc123:segment:00001" in repaired
    assert result.status == "succeeded"
