from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.analysis.narrative import build_narrative_prompts
from conference_pipeline.analysis.topic_reconciliation import (
    DEFAULT_QWEN36_MODEL,
    _compact_source_pack,
    build_reconciliation_prompt,
    collect_reconciliation_input,
    run_topic_reconciliation,
    strip_foreign_title_aliases,
    strip_unledgered_citations,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True))


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    topic_id = "09_efficiency_systems_architectures"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "run_id": "test-run",
            "topic_id": topic_id,
            "title": "Efficiency, systems, and architectures",
            "status": "deep-narrative-v1",
            "research_id": "test-run:09:abc",
            "research_path": "analysis/provenance/query_plans/09.json",
            "finding_text": "Efficiency evidence spans papers, talks, and deep dives.",
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "extraction_record_id": "record-1",
                    "source_artifact_path": "storage/source.txt",
                    "quote": "Efficient inference depends on batching and memory pressure.",
                },
                {
                    "chunk_id": "chunk-2",
                    "extraction_record_id": "record-2",
                    "source_artifact_path": "storage/source-2.txt",
                    "quote": "Systems reports compare throughput and quality.",
                },
                {
                    "chunk_id": "chunk-3",
                    "extraction_record_id": "record-3",
                    "source_artifact_path": "storage/source-3.txt",
                    "quote": "Architectural changes alter the deployment envelope.",
                },
            ],
            "llm_full_read_evidence": {
                "status": "succeeded",
                "unit_count": 2,
                "input_kind_counts": {"transcript_segment": 1, "paper_abstract": 1},
                "selected_items": [
                    {"unit_id": "u1", "summary": "Talk summary."},
                    {"unit_id": "u2", "summary": "Paper summary."},
                ],
            },
            "matched_papers": [],
            "extraction_record_ids": ["record-1", "record-2", "record-3", "u1", "u2"],
        },
    )
    _write(
        workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.md",
        "# Papers\n\nInference efficiency papers emphasize memory and latency.\n",
    )
    _write(
        workspace
        / "analysis"
        / "exhaustive_summaries"
        / f"{topic_id}__transcripts.md",
        "# Transcripts\n\nTalks emphasize production constraints.\n",
    )
    # Deep-dive ids truncate topic ids to the selected-lane prefix length.
    _write(
        workspace
        / "analysis"
        / "deep_dive_topics"
        / "09_efficiency_systems_architectu__dd01_under_realistic_long_co__global.md",
        "# Global\n\nGlobal retrieval says batching dominates.\n",
    )
    _write(
        workspace
        / "analysis"
        / "deep_dive_topics"
        / "09_efficiency_systems_architectu__dd01_under_realistic_long_co__heuristic.md",
        "# Heuristic\n\nKeyword retrieval says KV-cache memory dominates.\n",
    )
    return workspace


class _FakeResearcher:
    model = DEFAULT_QWEN36_MODEL

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.provider_receipt_paths = ["before.json"]

    async def run(self, query: str, *, context_path: Path | None = None) -> str:
        self.prompts.append(query)
        self.provider_receipt_paths.append("after.json")
        return (
            "# Topic Evidence Reconciliation\n\n"
            "## Consensus Thesis\n\n"
            "Efficiency evidence reconciles papers, talks, and deep dives.\n\n"
            "## Evidence Map\n\n"
            "### Papers\n\nPaper evidence.\n\n"
            "### Transcripts And Talks\n\nTranscript evidence.\n\n"
            "### Deep-Dive Research\n\nDeep-dive evidence.\n\n"
            "## Agreements Across Sources\n\nThey agree.\n\n"
            "## Tensions And Disagreements\n\nThey differ on bottlenecks.\n\n"
            "## Strongest Slide-Worthy Claims\n\n- Memory is the story.\n\n"
            "## Evidence To Preserve In Final Narrative\n\n- Keep ids.\n\n"
            "## Caveats And Source Gaps\n\n- More measurement needed.\n\n"
            "## Recommended Final Narrative Emphasis\n\nLead with deployment.\n"
        )

    def usage(self) -> dict[str, object]:
        return {"request_count": len(self.prompts)}


def test_topic_reconciliation_collects_summary_source_pack_and_dual_deep_dives(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    inputs = collect_reconciliation_input(
        workspace,
        topic_id="09_efficiency_systems_architectures",
    )

    assert inputs.input_token_estimate > 0
    assert len(inputs.deep_dive_paths) == 2
    assert "Exhaustive Paper Summary" in inputs.prompt
    assert "Global retrieval says batching dominates" in inputs.prompt
    assert "Keyword retrieval says KV-cache memory dominates" in inputs.prompt


def test_live_topic_reconciliation_enriches_source_pack_for_final_narrative(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    researcher = _FakeResearcher()

    receipt = run_topic_reconciliation(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        topics=["09_efficiency_systems_architectures"],
        researcher=researcher,
        resume=True,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["local_provider_receipt_paths"] == ["after.json"]
    assert receipt["source_packs_enriched"] is True
    source_pack = json.loads(
        (
            workspace
            / "analysis"
            / "narrative_source_packs"
            / "09_efficiency_systems_architectures.json"
        ).read_text(encoding="utf-8")
    )
    reconciliation = source_pack["topic_reconciliation"]
    assert reconciliation["status"] == "succeeded"
    assert reconciliation["deep_dive_report_count"] == 2
    assert "Efficiency evidence reconciles" in reconciliation["reconciliation_text"]
    _, prompt = build_narrative_prompts(source_pack)
    assert "## Topic evidence reconciliation" in prompt
    assert "Efficiency evidence reconciles" in prompt


def test_compact_source_pack_canonicalizes_raw_corpus_prefixed_ids() -> None:
    """CVPR-style raw ids (paper:/workshop_paper:/event_media_caption:) must
    never reach the reconciliation/repair prompt in raw form -- the model is
    instructed to preserve visible ids, and a raw id leak was confirmed to
    produce spurious unknown-reference validator blockers (SOURCE_ID_TOKEN_RE
    truncates slash-containing raw workshop-paper ids into garbage tokens)."""

    source_pack = {
        "topic_id": "01_multimodal_foundation_models",
        "title": "Multimodal Foundation Models",
        "chunks": [
            {
                "chunk_id": "event_media_caption:abc123:0",
                "extraction_record_id": "event_media_caption:abc123",
                "quote": "A talk about multimodal grounding.",
            },
            {
                "chunk_id": "workshop_paper:host.com/a/b:PID001:0",
                "extraction_record_id": "workshop_paper:host.com/a/b:PID001",
                "quote": "Workshop paper abstract.",
            },
        ],
        "llm_full_read_evidence": {
            "selected_items": [
                {"unit_id": "event_media_caption:abc123:segment:00000"},
            ]
        },
        "matched_papers": [],
        "extraction_record_ids": [
            "event_media_caption:abc123",
            "workshop_paper:host.com/a/b:PID001",
        ],
    }

    compact = _compact_source_pack(source_pack)

    chunk_ids = [chunk["chunk_id"] for chunk in compact["chunks"]]
    extraction_ids = [chunk["extraction_record_id"] for chunk in compact["chunks"]]
    for raw_id in chunk_ids + extraction_ids + compact["extraction_record_ids"]:
        assert not raw_id.startswith("event_media_caption:"), raw_id
        assert not raw_id.startswith("workshop_paper:"), raw_id
    assert "speech_transcript:abc123" in chunk_ids
    assert "paper_abstract:host.com_a_b_PID001" in chunk_ids
    assert (
        compact["llm_full_read_evidence"]["selected_items"][0]["unit_id"]
        == "speech_transcript:abc123:segment:00000"
    )

    prompt = build_reconciliation_prompt(
        topic_id="01_multimodal_foundation_models",
        title="Multimodal Foundation Models",
        source_pack=source_pack,
        paper_summary_text="",
        transcript_summary_text="",
        deep_dive_texts={},
    )
    assert "event_media_caption:" not in prompt
    assert "workshop_paper:" not in prompt
    assert "Cite ONLY the canonical ids" in prompt


def test_rerun_reconciliation_with_new_markdown_makes_its_validation_stale(tmp_path: Path) -> None:
    """07E binds 07D by the content of 07D's receipt. A re-run writing different
    reconciliation Markdown with the same topics and counts must change that
    content (via ``substep_outputs``), so the old validation reads stale."""

    from conference_pipeline import receipt_lineage

    workspace = _workspace(tmp_path)
    run_id = "test-run"
    first = run_topic_reconciliation(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        topics=["09_efficiency_systems_architectures"],
        researcher=_FakeResearcher(),
        resume=False,
    )
    assert first["status"] == "succeeded"
    assert first["lineage"]["binding_source"]["substep_id"] == "topic-reconciliation"
    assert first["substep_outputs"]["artifacts"][0]["path"] == "analysis/topic_reconciliations"

    validation = receipt_lineage.stamp_substep_receipt(
        {"status": "succeeded"}, workspace=workspace, run_id=run_id, substep_id="topic-reference-validation"
    )
    assert receipt_lineage.verify_lineage(workspace, run_id, validation["lineage"], stage_id="07E", transitive=False) == []

    # An unchanged resume (07D reuses the topic, rewriting only its "reused"
    # bookkeeping) must not make the validation stale.
    unchanged_researcher = _FakeResearcher()
    resumed = run_topic_reconciliation(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        topics=["09_efficiency_systems_architectures"],
        researcher=unchanged_researcher,
        resume=True,
    )
    assert resumed["status"] == "succeeded"
    assert unchanged_researcher.prompts == [], "unchanged inputs: the topic is reused"
    assert resumed["substep_outputs"]["digest"] == first["substep_outputs"]["digest"]
    assert receipt_lineage.verify_lineage(
        workspace, run_id, validation["lineage"], stage_id="07E", transitive=False
    ) == []

    class _RevisedResearcher(_FakeResearcher):
        async def run(self, query: str, *, context_path: Path | None = None) -> str:
            text = await super().run(query, context_path=context_path)
            return text.replace("Memory is the story.", "Batching is the story.")

    second = run_topic_reconciliation(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        topics=["09_efficiency_systems_architectures"],
        researcher=_RevisedResearcher(),
        resume=False,
    )
    assert second["status"] == "succeeded"
    assert second["completed_topic_count"] == first["completed_topic_count"]
    blockers = receipt_lineage.verify_lineage(workspace, run_id, validation["lineage"], stage_id="07E", transitive=False)
    assert [(b["code"], b["upstream_stage_id"]) for b in blockers] == [
        ("stage_contract_upstream_changed", "topic-reconciliation")
    ]


def test_repaired_reconciliation_survives_an_unchanged_resume(tmp_path: Path) -> None:
    """07F/07G repair 07D's Markdown in place and refresh its output digest; the
    validation bound after the repair stays fresh through a zero-call 07D
    resume that re-derives the sidecar's Markdown hashes."""

    from conference_pipeline import receipt_lineage

    workspace = _workspace(tmp_path)
    run_id = "test-run"
    topic = "09_efficiency_systems_architectures"
    run_topic_reconciliation(
        workspace=workspace, run_id=run_id, mode="live", topics=[topic], researcher=_FakeResearcher(), resume=False
    )
    markdown = workspace / "analysis" / "topic_reconciliations" / f"{topic}.md"
    markdown.write_text(markdown.read_text(encoding="utf-8") + "\n- Repaired citation.\n", encoding="utf-8")
    assert receipt_lineage.refresh_substep_outputs(workspace, run_id, "topic-reconciliation") is True
    receipt_path = workspace / "storage" / "workflow-runs" / run_id / "07D-topic-reconciliation.json"
    repaired_digest = json.loads(receipt_path.read_text(encoding="utf-8"))["substep_outputs"]["digest"]
    validation = receipt_lineage.stamp_substep_receipt(
        {"status": "succeeded"}, workspace=workspace, run_id=run_id, substep_id="topic-reference-validation"
    )

    researcher = _FakeResearcher()
    resumed = run_topic_reconciliation(
        workspace=workspace, run_id=run_id, mode="live", topics=[topic], researcher=researcher, resume=True
    )
    assert researcher.prompts == []
    assert "Repaired citation" in markdown.read_text(encoding="utf-8")
    assert resumed["substep_outputs"]["digest"] == repaired_digest
    assert receipt_lineage.verify_lineage(
        workspace, run_id, validation["lineage"], stage_id="07E", transitive=False
    ) == []


def test_dry_run_reconciliation_reports_succeeded_but_is_never_bound(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    receipt = run_topic_reconciliation(
        workspace=workspace, run_id="test-run", mode="dry-run", topics=["09_efficiency_systems_architectures"]
    )
    assert receipt["mode"] == "dry-run"
    assert "lineage" not in receipt and "substep_outputs" not in receipt
    assert receipt["status"] == "succeeded" and receipt["lineage_unbound_reason"] == "mode:dry-run"


def test_live_resume_never_reuses_a_dry_run_placeholder(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    topic = "09_efficiency_systems_architectures"
    run_topic_reconciliation(workspace=workspace, run_id="test-run", mode="dry-run", topics=[topic])
    researcher = _FakeResearcher()
    live = run_topic_reconciliation(
        workspace=workspace, run_id="test-run", mode="live", topics=[topic], researcher=researcher, resume=True
    )
    assert researcher.prompts, "a dry-run placeholder must be synthesized for real on a live resume"
    assert live["lineage"]["binding_source"]["substep_id"] == "topic-reconciliation"


def test_reconciliation_lineage_binds_the_source_packs_it_read(tmp_path: Path) -> None:
    """Source packs are inputs to 07D/07E with no upstream receipt of their own:
    a changed pack makes 07D stale, while 07D/07E's own enrichment of the pack
    (``topic_reconciliation``) and restamped receipt digests do not."""

    from conference_pipeline import receipt_lineage

    workspace = _workspace(tmp_path)
    topic = "09_efficiency_systems_architectures"
    receipt = run_topic_reconciliation(
        workspace=workspace, run_id="test-run", mode="live", topics=[topic], researcher=_FakeResearcher(), resume=False
    )
    assert receipt["lineage"]["input_artifacts"]["narrative_source_packs"]["path"] == "analysis/narrative_source_packs"

    def input_blockers() -> list[str]:
        return [
            b["code"]
            for b in receipt_lineage.verify_input_artifacts(workspace, receipt["lineage"], stage_id="07D")
        ]

    assert input_blockers() == []  # 07D's own enrichment of the pack is excluded
    pack_path = workspace / "analysis" / "narrative_source_packs" / f"{topic}.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["topic_reconciliation"] = {**pack.get("topic_reconciliation", {}), "reference_validation": {"status": "ok"}}
    pack.setdefault("llm_full_read_evidence", {})["receipt_sha256"] = "e" * 64
    pack_path.write_text(json.dumps(pack), encoding="utf-8")
    assert input_blockers() == []
    pack["chunks"] = [*pack.get("chunks", []), {"chunk_id": "new", "text": "new evidence"}]
    pack_path.write_text(json.dumps(pack), encoding="utf-8")
    assert input_blockers() == ["stage_contract_input_artifact_changed"]


def test_strip_unledgered_citations_drops_fabricated_ids_and_keeps_valid_ones(
    tmp_path: Path,
) -> None:
    """The model fabricates transcript ids for deep-dive quotes.

    The deep-dive reports quote transcripts without carrying their ledger
    ids, so the reconciliation model invents a plausible-looking id to keep
    the quote -- observed live on CVPR 2026 topic 12, which produced
    ``speech_transcript:q7Wv5A_lOFw`` ("inferred from deep dive") and a
    ``lane__video`` id where the corpus joins with ``lane:video``. The
    prompt already forbids this and the model still does it, so the guard
    has to be deterministic. It must drop exactly the unbindable citations
    and leave every valid one untouched.
    """

    from conference_pipeline.analysis.topic_reference_validation import (
        build_reference_ledger,
        validate_reconciliation_references,
    )

    workspace = tmp_path / "conf"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / "01_topic.json",
        {
            "topic_id": "01_topic",
            "title": "Topic One",
            "chunks": [
                {
                    "chunk_id": "event_media_caption:realvid:0",
                    "extraction_record_id": "event_media_caption:realvid",
                    "quote": "A real indexed talk.",
                    "talk_title": "A Real Talk",
                }
            ],
            "llm_full_read_evidence": {"selected_items": []},
            "matched_papers": [],
            "extraction_record_ids": ["event_media_caption:realvid"],
        },
    )
    ledger = build_reference_ledger(workspace, "01_topic")

    text = (
        "# Topic Evidence Reconciliation\n"
        "## Evidence Map\n"
        "*   `speech_transcript:realvid`: a genuinely bound talk.\n"
        '*   > "Quote the model liked." (`speech_transcript:madeup1` '
        "- inferred from deep dive)\n"
        "*   `speech_transcript:lane__madeup2:segment:00000`: (No summary provided).\n"
    )

    before = validate_reconciliation_references(
        text=text, ledger=ledger, require_citations=False
    )
    assert before.unknown_reference_count == 2

    cleaned, removed = strip_unledgered_citations(text, ledger=ledger)

    after = validate_reconciliation_references(
        text=cleaned, ledger=ledger, require_citations=False
    )
    assert after.unknown_reference_count == 0
    # The bound citation survives; only the fabricated ones go.
    assert "speech_transcript:realvid" in cleaned
    assert "madeup1" not in cleaned
    assert "madeup2" not in cleaned
    assert set(after.valid_reference_ids) == set(before.valid_reference_ids)

    # Removals are reported, never silent.
    assert len(removed) == 2
    assert {entry["action"] for entry in removed} == {"dropped_list_item"}
    assert sorted(
        reference_id for entry in removed for reference_id in entry["reference_ids"]
    ) == ["speech_transcript:lane__madeup2:segment:00000", "speech_transcript:madeup1"]


def test_strip_unledgered_citations_is_a_no_op_when_every_citation_binds(
    tmp_path: Path,
) -> None:
    """A clean reconciliation must come back byte-identical.

    The guard runs on every fresh synthesis, so it has to be inert for the
    eleven topics that were already passing -- otherwise it would rewrite
    good output and churn their downstream digests.
    """

    from conference_pipeline.analysis.topic_reference_validation import (
        build_reference_ledger,
    )

    workspace = tmp_path / "conf"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / "01_topic.json",
        {
            "topic_id": "01_topic",
            "title": "Topic One",
            "chunks": [
                {
                    "chunk_id": "event_media_caption:realvid:0",
                    "extraction_record_id": "event_media_caption:realvid",
                    "quote": "A real indexed talk.",
                }
            ],
            "llm_full_read_evidence": {"selected_items": []},
            "matched_papers": [],
            "extraction_record_ids": ["event_media_caption:realvid"],
        },
    )
    ledger = build_reference_ledger(workspace, "01_topic")
    text = "# Topic Evidence Reconciliation\n*   `speech_transcript:realvid`: bound.\n"

    cleaned, removed = strip_unledgered_citations(text, ledger=ledger)

    assert cleaned == text
    assert removed == []


def test_strip_foreign_title_aliases_keeps_the_citation_and_the_indent(
    tmp_path: Path,
) -> None:
    """A borrowed label is not a bad citation.

    Live CVPR 2026 topic 02 produced
    ``speech_transcript:AqneAlYzEAs:segment:00000 (PSIVG)`` -- a correctly
    bound transcript tagged with a short name belonging to a different
    ledger entry. Dropping the line (what the unledgered-citation guard
    does) would throw away a good reference, so only the label goes. The
    leading indent must survive too: collapsing it re-levels the markdown
    list the line sits in.
    """

    from conference_pipeline.analysis.topic_reference_validation import (
        build_reference_ledger,
        validate_reconciliation_references,
    )

    workspace = tmp_path / "conf"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / "01_topic.json",
        {
            "topic_id": "01_topic",
            "title": "Topic One",
            "chunks": [
                {
                    "chunk_id": "event_media_caption:goodvid:0",
                    "extraction_record_id": "event_media_caption:goodvid",
                    "talk_title": "Neural Rendering At Scale",
                    "quote": "A bound talk.",
                },
                {
                    "chunk_id": "event_media_caption:othervid:0",
                    "extraction_record_id": "event_media_caption:othervid",
                    "talk_title": "Quantum Sensing Primer",
                    "quote": "A different bound talk.",
                },
            ],
            "llm_full_read_evidence": {"selected_items": []},
            "matched_papers": [],
            "extraction_record_ids": [
                "event_media_caption:goodvid",
                "event_media_caption:othervid",
            ],
        },
    )
    ledger = build_reference_ledger(workspace, "01_topic")

    # Cites goodvid but labels it with the OTHER entry's title.
    text = (
        "# Topic Evidence Reconciliation\n"
        "    *   *Source:* `speech_transcript:goodvid` (Quantum Sensing Primer); more.\n"
    )
    before = validate_reconciliation_references(
        text=text, ledger=ledger, require_citations=False
    )
    assert before.title_mismatch_count == 1

    cleaned, removed = strip_foreign_title_aliases(text, ledger=ledger)

    after = validate_reconciliation_references(
        text=cleaned, ledger=ledger, require_citations=False
    )
    assert after.title_mismatch_count == 0
    # The citation survives; only the borrowed label is gone.
    assert "speech_transcript:goodvid" in cleaned
    assert "Quantum Sensing Primer" not in cleaned
    assert set(after.valid_reference_ids) == set(before.valid_reference_ids)
    # The markdown list level is unchanged.
    assert cleaned.splitlines()[1].startswith("    ")
    assert len(removed) == 1
    assert removed[0]["action"] == "stripped_foreign_title_alias"


def test_strip_foreign_title_aliases_is_a_no_op_when_labels_match(
    tmp_path: Path,
) -> None:
    """A correctly labelled citation must come back byte-identical."""

    from conference_pipeline.analysis.topic_reference_validation import (
        build_reference_ledger,
    )

    workspace = tmp_path / "conf"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / "01_topic.json",
        {
            "topic_id": "01_topic",
            "title": "Topic One",
            "chunks": [
                {
                    "chunk_id": "event_media_caption:goodvid:0",
                    "extraction_record_id": "event_media_caption:goodvid",
                    "talk_title": "Neural Rendering At Scale",
                    "quote": "A bound talk.",
                }
            ],
            "llm_full_read_evidence": {"selected_items": []},
            "matched_papers": [],
            "extraction_record_ids": ["event_media_caption:goodvid"],
        },
    )
    ledger = build_reference_ledger(workspace, "01_topic")
    text = (
        "# Topic Evidence Reconciliation\n"
        "    *   `speech_transcript:goodvid` (Neural Rendering At Scale).\n"
    )

    cleaned, removed = strip_foreign_title_aliases(text, ledger=ledger)

    assert cleaned == text
    assert removed == []


def test_strip_foreign_title_aliases_never_edits_inside_a_backticked_id(
    tmp_path: Path,
) -> None:
    """Live CVPR 2026 topic 08: the alias "workshop" was excised from
    `paper_abstract:thecvf.com_CVPR_2026_Workshop_EDGE_sbjLkIbvsW`, turning a
    valid citation into `..._CVPR_2026__EDGE_...`, an unknown id."""

    from conference_pipeline.analysis.topic_reference_validation import (
        build_reference_ledger,
        validate_reconciliation_references,
    )

    workspace = tmp_path / "conf"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / "01_topic.json",
        {
            "topic_id": "01_topic",
            "title": "Topic One",
            "chunks": [
                {
                    "chunk_id": "workshop_paper:thecvf.com/CVPR/2026/Workshop/EDGE:sbjLkIbvsW:0",
                    "extraction_record_id": "workshop_paper:thecvf.com/CVPR/2026/Workshop/EDGE:sbjLkIbvsW",
                    "paper_title": "A KL Lens On Quantization",
                    "quote": "Fast sensitivity for mixed precision.",
                },
                {
                    "chunk_id": "event_media_caption:othervid:0",
                    "extraction_record_id": "event_media_caption:othervid",
                    "talk_title": "Edge Deployment Panel Discussion",
                    "quote": "A different bound talk.",
                },
            ],
            "llm_full_read_evidence": {"selected_items": []},
            "matched_papers": [],
            "extraction_record_ids": [
                "workshop_paper:thecvf.com/CVPR/2026/Workshop/EDGE:sbjLkIbvsW",
                "event_media_caption:othervid",
            ],
        },
    )
    ledger = build_reference_ledger(workspace, "01_topic")
    cited = "`paper_abstract:thecvf.com_CVPR_2026_Workshop_EDGE_sbjLkIbvsW`"
    # The line borrows the other entry's title as a label.
    text = f"# R\n    *   {cited} (Edge Deployment Panel Discussion): fast sensitivity.\n"
    before = validate_reconciliation_references(text=text, ledger=ledger, require_citations=False)
    assert before.title_mismatch_count == 1

    cleaned, _removed = strip_foreign_title_aliases(text, ledger=ledger)

    assert cited in cleaned
    after = validate_reconciliation_references(text=cleaned, ledger=ledger, require_citations=False)
    assert after.unknown_reference_count == 0
    assert after.title_mismatch_count == 0

    # And a bare alias phrase inside the id token is untouched even when the
    # parenthetical route does not apply.
    text2 = f"# R\n    *   {cited} Edge Deployment Panel Discussion says so.\n"
    cleaned2, _ = strip_foreign_title_aliases(text2, ledger=ledger)
    assert cited in cleaned2
    assert "Edge Deployment Panel Discussion" not in cleaned2
