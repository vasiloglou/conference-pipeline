"""Recovering papers that were blocked from the visible-source catalog by
bookkeeping rather than by missing evidence.

Live discovery (ICML 2026 topic 01): the blueprint needs 25 distinct works and
the topic offered 17, so the deck could not be built. None of the 28 blocked
ledger entries lacked real evidence:

* 5 papers arrived from candidate packets titled ``"paper"`` -- a kind noun the
  generic-title check did not recognise -- so they never reached the OpenReview
  title enrichment and collapsed into one catalog work;
* those same packets were flagged ``packet_truncated`` because a count cap of
  12 cut a 16-candidate list, while the packet measured ~4.2k tokens against an
  8k budget, and that flag blocked every candidate in them;
* 8 titled papers had no contribution note even though the topic's own
  narrative source pack recorded an evidence snippet for each.

The 12 genuinely ambiguous candidates (top two scores within the margin) stay
blocked: that is referent ambiguity, not bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.analysis.topic_reference_candidate_resolution import (
    CandidateResolutionConfig,
    _fit_candidates_to_token_budget,
)
from conference_pipeline.presentation.source_inventory import (
    _enrich_paper_contribution_notes,
    _first_sentences,
    _is_generic_source_title,
    _packet_blocked_status,
    visible_source_chip_eligible,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def test_kind_noun_titles_are_placeholders_but_real_titles_survive() -> None:
    for placeholder in ("paper", "Paper", "papers", "poster", "talk", "abstract", "work", "source", "workshop paper"):
        assert _is_generic_source_title(placeholder), placeholder
    for title in (
        "The Paper Trail: Auditing Agent Provenance",
        "Poster Child: Rethinking Benchmarks",
        "Talking Heads Are Not Enough",
        "Work Stealing for Distributed Training",
    ):
        assert not _is_generic_source_title(title), title


def test_first_sentences_trims_on_a_sentence_boundary() -> None:
    text = (
        "NOVA-Test asks whether an LLM-generated hypothesis is falsifiable. It scores each "
        "hypothesis on five axes and reports a falsification plan for every claim it accepts, "
        "which the authors argue is the missing piece in automated science pipelines today."
    )
    out = _first_sentences(text)
    assert len(out) <= 240
    assert out.endswith(".") and "..." not in out
    assert _first_sentences("Short note.") == "Short note."
    assert _first_sentences("") == ""


def test_count_cap_truncation_under_budget_no_longer_blocks_candidates() -> None:
    bookkeeping = {
        "packet_truncated": True,
        "token_estimate": 4227,
        "config": {"max_packet_input_tokens": 8000, "max_candidates_per_reference": 12},
    }
    assert _packet_blocked_status(bookkeeping) is None

    budget_bound = {**bookkeeping, "token_estimate": 7800}
    assert _packet_blocked_status(budget_bound) == "packet_too_large"

    # Referent ambiguity is evidence, not bookkeeping: it still blocks.
    ambiguous = {
        "packet_truncated": False,
        "token_estimate": 5109,
        "config": {"max_packet_input_tokens": 8000},
        "blockers": [{"code": "candidate_resolution_ambiguous"}],
    }
    assert _packet_blocked_status(ambiguous) == "ambiguous"


def test_paper_contribution_notes_come_from_workspace_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "storage/extractions/run/llm-full-read/by-unit/paper_abstract_AAA.json",
        {"summary": "SciPredict benchmarks whether models can predict experimental outcomes."},
    )
    _write(
        workspace / "analysis/narrative_source_packs/01_topic.json",
        {
            "matched_papers": [
                {"paper_id": "BBB", "title": "NOVA-Test", "evidence_snippet": "NOVA-Test scores LLM hypotheses for falsifiability."}
            ]
        },
    )
    _write(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {"papers": [{"id": "CCC", "title": "Third Paper", "abstract": "We introduce a solver for stiff PDEs. It converges fast."}]},
    )
    drafts = {
        f"paper_abstract:{pid}": {
            "source_id": f"paper_abstract:{pid}",
            "resolved_title": title,
            "works": [title],
            "authors": ["An Author"],
            "speakers": [],
            "contribution_note": "",
            "confidence": 1.0,
            "status": "missing_contribution",
        }
        for pid, title in (("AAA", "SciPredict"), ("BBB", "NOVA-Test"), ("CCC", "Third Paper"), ("DDD", "Unknown Work"))
    }
    drafts["paper_abstract:EEE"] = {
        "source_id": "paper_abstract:EEE",
        "resolved_title": "Already Described",
        "works": ["Already Described"],
        "authors": ["An Author"],
        "speakers": [],
        "contribution_note": "Existing note stays.",
        "confidence": 1.0,
        "status": "resolved",
    }

    enriched = _enrich_paper_contribution_notes(workspace, drafts, topic_id="01_topic")

    assert enriched == 3
    assert drafts["paper_abstract:AAA"]["contribution_note"].startswith("SciPredict benchmarks")
    assert drafts["paper_abstract:BBB"]["contribution_note"].startswith("NOVA-Test scores")
    assert drafts["paper_abstract:CCC"]["contribution_note"].startswith("We introduce a solver")
    assert drafts["paper_abstract:DDD"]["contribution_note"] == ""  # nothing invented
    assert drafts["paper_abstract:EEE"]["contribution_note"] == "Existing note stays."
    assert visible_source_chip_eligible(drafts["paper_abstract:AAA"])
    assert not visible_source_chip_eligible(drafts["paper_abstract:DDD"])


def test_candidate_packet_grows_to_the_real_token_budget() -> None:
    config = CandidateResolutionConfig(max_candidates_per_reference=12, max_packet_input_tokens=8000)
    candidates = [f"cand-{i:03d}" for i in range(16)]
    # ~350 tokens per candidate: 16 candidates fit inside the 8000-token budget.
    kept = _fit_candidates_to_token_budget(
        candidates, config=config, payload_for=lambda rows: {"c": ["x" * 1400 for _ in rows]}
    )
    assert len(kept) == 16

    # A heavy packet stops where the budget really binds, keeping at least the cap.
    kept_heavy = _fit_candidates_to_token_budget(
        candidates, config=config, payload_for=lambda rows: {"c": ["x" * 9000 for _ in rows]}
    )
    assert len(kept_heavy) == 12

    ceiling = CandidateResolutionConfig(max_candidates_per_reference=2, max_candidate_hard_ceiling=4)
    kept_ceiling = _fit_candidates_to_token_budget(
        candidates, config=ceiling, payload_for=lambda rows: {"c": list(rows)}
    )
    assert len(kept_ceiling) == 4
