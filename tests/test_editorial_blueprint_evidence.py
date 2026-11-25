from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_editorial_blueprint_module():
    module_path = PROJECT_ROOT / "scripts" / "presentations" / "generate_editorial_blueprint.py"
    spec = importlib.util.spec_from_file_location("generate_editorial_blueprint_test", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evidence_digest_includes_full_analysis_markdown(tmp_path):
    module = _load_editorial_blueprint_module()
    analysis = tmp_path / "analysis.md"
    analysis.write_text("A" * 20_000 + "TAIL_MARKER", encoding="utf-8")

    digest = module.build_evidence_digest(analysis, None)

    assert "## Topic analysis\n" in digest
    assert "TAIL_MARKER" in digest


def test_editorial_blueprint_prompts_include_full_evidence(monkeypatch):
    module = _load_editorial_blueprint_module()
    captured_prompts: list[str] = []

    def fake_call(prompt, *, max_tokens=8000):
        captured_prompts.append(prompt)
        if "Author the editorial plan" in prompt:
            return {"selected_sources": [], "clusters": [], "sections": []}
        return []

    monkeypatch.setattr(module, "_call", fake_call)

    module.author_board_and_spine(
        topic_title="Demo",
        catalog=[],
        evidence="A" * 17_000 + "BOARD_TAIL_MARKER",
    )
    module.author_section_slides(
        topic_title="Demo",
        section={"title": "Section", "intent": "Intent", "slide_budget": 1},
        section_sources=[],
        evidence="B" * 13_000 + "SECTION_TAIL_MARKER",
        prior_titles=[],
    )

    assert "BOARD_TAIL_MARKER" in captured_prompts[0]
    assert "SECTION_TAIL_MARKER" in captured_prompts[1]


# ---------------------------------------------------------------------------
# Phase 10.3 Plan 03 — the pseudo-people guard (V-07 positive, V-08 control)
#
# `_people_from_title` regexes capitalised phrases out of a title and presents
# them as human beings whenever an entry has none.  It is reached from TWO
# places and both matter: `build_source_catalog` (which sets the catalog people
# CEILING) and `_mention_payload` (which writes pseudo-SPEAKERS straight into
# the blueprint mentions the coverage statistics count for the post-LLM people
# gate).  Landing real event-media titles unguarded was measured at 341 distinct
# fabricated people across the twelve CVPR decks; the author join alone cuts
# that to 23, not to zero, because 19 of the 181 deck videos have no author join
# to give.  Only the guard closes it.
# ---------------------------------------------------------------------------

# CROSS-PLAN CONTRACT.  Plan 04's enricher WRITES exactly this key and this
# value onto every ledger row whose title came from the deterministic local
# join; this plan's guard READS exactly these at both call sites.  The two plans
# run in different waves and cannot discover the contract from each other at
# runtime, so it is declared here and recorded in both SUMMARY files.
PROVENANCE_MARKER_KEY = "resolved_title_kind"
PROVENANCE_MARKER_VALUE = "local_join"

# The additive, restrictive conjunct the guard appends to the parser-fallback
# condition at each call site.  It must be byte-identical at both sites: the
# mutation harness below locates it by exact substring and drops one occurrence
# at a time, and it asserts the count is exactly two so a future refactor that
# collapses or duplicates the guard turns this test red instead of silently
# weakening coverage.
GUARD_CONJUNCT = ' and entry.get("resolved_title_kind") != "local_join"'

# `Oral Session` is the single most frequent fabrication measured against the
# real CVPR event-media titles (x6 among the 341 distinct pseudo-people that
# real titles would introduce unguarded), so this fixture pins an observed
# hazard rather than a contrived one.  Verified live: the unguarded parser
# returns exactly ['Oral Session'] for it.
FIXTURE_TITLE = "Oral Session, Hall A"
FIXTURE_PSEUDO_PERSON = "Oral Session"
FIXTURE_SOURCE_ID = "speech_transcript:aBcD1234:segment:00000"

_MARKER_ABSENT = object()


def _guard_fixture_entry(marker=_MARKER_ABSENT):
    """A chip-eligible, people-less ledger row carrying a real event title.

    ``marker`` absent means the row carries no provenance marker at all; passing
    a value writes it under the contract key.
    """
    entry = {
        "source_id": FIXTURE_SOURCE_ID,
        "visible_source_chip_eligible": True,
        "resolved_title": FIXTURE_TITLE,
        "authors": [],
        "speakers": [],
        "contribution_note": "Transcript row enriched by the deterministic local join.",
        "resolution_status": "resolved",
        "source_family": "artifacts",
        "confidence": 0.9,
    }
    if marker is not _MARKER_ABSENT:
        entry[PROVENANCE_MARKER_KEY] = marker
    return entry


def _catalog_row(module, entry):
    catalog = module.build_source_catalog({"entries": [entry]})
    assert len(catalog) == 1, catalog
    return catalog[0]


def _mention(module, entry):
    payload = module._mention_payload(
        FIXTURE_SOURCE_ID, {FIXTURE_SOURCE_ID: entry}, module.LabelRegistry()
    )
    assert payload is not None
    return payload


def _load_module_with_guard_dropped(tmp_path, occurrence: int, module_name: str):
    """Load a copy of the generator with ONE guard occurrence deleted.

    ``occurrence`` 0 is the catalog site and 1 is the mention site, because
    ``build_source_catalog`` precedes ``_mention_payload`` in the file.  Fails
    loudly if the conjunct is not present exactly twice, so the drop-proof can
    never pass by mutating nothing.
    """
    source_path = PROJECT_ROOT / "scripts" / "presentations" / "generate_editorial_blueprint.py"
    src = source_path.read_text(encoding="utf-8")
    found = src.count(GUARD_CONJUNCT)
    assert found == 2, (
        f"expected the guard conjunct {GUARD_CONJUNCT!r} exactly twice "
        f"(build_source_catalog then _mention_payload); found {found}. "
        "The guard is missing, was collapsed into one site, or was duplicated."
    )
    head, middle, tail = src.split(GUARD_CONJUNCT)
    if occurrence == 0:
        mutated = head + middle + GUARD_CONJUNCT + tail
    else:
        mutated = head + GUARD_CONJUNCT + middle + tail
    assert mutated != src and mutated.count(GUARD_CONJUNCT) == 1

    # Mirror the real file's depth so the module's own ``parents[2]`` resolution
    # stays well-formed, and make sure the real ``src/`` tree is importable for
    # the module's package-level imports.
    real_src = str(PROJECT_ROOT / "src")
    if real_src not in sys.path:
        sys.path.insert(0, real_src)
    mutated_dir = tmp_path / module_name / "scripts" / "presentations"
    mutated_dir.mkdir(parents=True, exist_ok=True)
    mutated_path = mutated_dir / "generate_editorial_blueprint.py"
    mutated_path.write_text(mutated, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(module_name, mutated_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_people_from_title_is_suppressed_for_locally_joined_rows(tmp_path):
    """V-07 — the guard, drop-proven independently at each of the two sites."""
    module = _load_editorial_blueprint_module()

    row = _catalog_row(module, _guard_fixture_entry(PROVENANCE_MARKER_VALUE))
    assert row["people"] == [], row
    assert row["has_named_people"] is False, row

    payload = _mention(module, _guard_fixture_entry(PROVENANCE_MARKER_VALUE))
    assert payload["speakers"] == [], payload

    # Drop the CATALOG guard only: the fabrication must reappear there, and the
    # mention site must still be protected.
    dropped_catalog = _load_module_with_guard_dropped(
        tmp_path, 0, "geb_guard_dropped_catalog"
    )
    leaked_row = _catalog_row(dropped_catalog, _guard_fixture_entry(PROVENANCE_MARKER_VALUE))
    assert leaked_row["people"] == [FIXTURE_PSEUDO_PERSON], leaked_row
    assert leaked_row["has_named_people"] is True, leaked_row
    still_guarded = _mention(dropped_catalog, _guard_fixture_entry(PROVENANCE_MARKER_VALUE))
    assert still_guarded["speakers"] == [], still_guarded

    # Drop the MENTION guard only: pseudo-speakers must reappear in the mention
    # payload, and the catalog site must still be protected.
    dropped_mention = _load_module_with_guard_dropped(
        tmp_path, 1, "geb_guard_dropped_mention"
    )
    leaked_payload = _mention(dropped_mention, _guard_fixture_entry(PROVENANCE_MARKER_VALUE))
    assert leaked_payload["speakers"] == [FIXTURE_PSEUDO_PERSON], leaked_payload
    still_guarded_row = _catalog_row(
        dropped_mention, _guard_fixture_entry(PROVENANCE_MARKER_VALUE)
    )
    assert still_guarded_row["people"] == [], still_guarded_row


def test_people_from_title_still_invents_for_unmarked_rows():
    """V-08 — negative control: the fabrication mechanism is demonstrably live.

    Without this, the guard could silently degenerate into a no-op that passes
    because the parser stopped firing for some unrelated reason.
    """
    module = _load_editorial_blueprint_module()

    # An absent marker and a present-but-empty marker must behave identically:
    # the guard's contract is a truthy, exact ``local_join`` value.
    for marker in (_MARKER_ABSENT, ""):
        entry = _guard_fixture_entry(marker)
        row = _catalog_row(module, entry)
        assert row["people"] == [FIXTURE_PSEUDO_PERSON], (marker, row)
        assert row["has_named_people"] is True, (marker, row)

        payload = _mention(module, _guard_fixture_entry(marker))
        assert payload["speakers"] == [FIXTURE_PSEUDO_PERSON], (marker, payload)

        # The invented name is drawn from the title -- it is a machine-parsed
        # noun phrase presented as a named human researcher.
        assert FIXTURE_PSEUDO_PERSON in FIXTURE_TITLE


def test_title_normalization_preserves_words_for_explicit_repair() -> None:
    module = _load_editorial_blueprint_module()
    title = "Computational Imaging, Physical Modeling, and Vision for Society & Arts"
    assert module._cap_title(title) == title
    assert module._cap_title("A short, complete title.") == "A short, complete title."


def _reply(text, stop_reason="end", block_types=("text",)):
    from types import SimpleNamespace

    return SimpleNamespace(text=text, stop_reason=stop_reason, block_types=block_types)


def test_call_repairs_one_malformed_json_reply(monkeypatch):
    """A reply with invalid JSON gets one repair attempt instead of failing the deck."""
    module = _load_editorial_blueprint_module()
    replies = ['[{"title": "A "quoted" claim"}]', '[{"title": "A \\"quoted\\" claim"}]']
    sent: list[list] = []

    def fake_complete(task_id, **kwargs):
        sent.append(kwargs["messages"])
        return _reply(replies[len(sent) - 1])

    monkeypatch.setattr(module, "complete", fake_complete)
    assert module._call("prompt") == [{"title": 'A "quoted" claim'}]
    assert len(sent) == 2 and sent[1][-1]["content"].startswith("Your previous reply was not valid JSON")

    replies[:] = ["not json", "still not json"]
    sent.clear()
    import pytest

    with pytest.raises(ValueError, match="parseable JSON"):
        module._call("prompt")
    assert len(sent) == 2


def test_call_reports_why_a_reply_was_empty(monkeypatch):
    import pytest

    module = _load_editorial_blueprint_module()
    calls = []

    def fake_complete(task_id, **kwargs):
        calls.append(kwargs["model"])
        return _reply("", block_types=())

    monkeypatch.setattr(module, "complete", fake_complete)
    with pytest.raises(ValueError, match="stop_reason='end'"):
        module._call("prompt")
    assert calls == [module.MODEL_ID], "an empty reply is not retried as malformed JSON"


def test_a_refused_call_is_answered_once_by_the_registry_fallback_and_recorded(monkeypatch):
    """ICML 2026 deck 01: Opus's safety filter cut the molecule section off
    mid-JSON on every attempt; the registry fallback (Sonnet) wrote it."""
    import pytest

    module = _load_editorial_blueprint_module()
    module.REFUSAL_FALLBACKS.clear()
    fallback = module.default_fallback_model(module.TASK_ID)
    fallback_provider = module.default_fallback_provider(module.TASK_ID)
    calls = []

    def fake_complete(task_id, **kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        if kwargs["model"] == module.MODEL_ID:
            return _reply('[{"ti', stop_reason="refusal")
        return _reply('[{"title": "T"}]')

    monkeypatch.setattr(module, "complete", fake_complete)
    assert module._call("prompt") == [{"title": "T"}]
    assert calls == [(module.PROVIDER, module.MODEL_ID), (fallback_provider, fallback)]
    assert module.REFUSAL_FALLBACKS[0]["fallback_model"] == fallback
    assert module.REFUSAL_FALLBACKS[0]["fallback_provider"] == fallback_provider
    assert module.REFUSAL_FALLBACKS[0]["primary_partial_chars"] == 5

    monkeypatch.setattr(module, "complete", lambda task_id, **kwargs: _reply("", stop_reason="refusal", block_types=()))
    with pytest.raises(ValueError, match="stop_reason='refusal'"):
        module._call("prompt")


def test_the_blueprint_route_can_be_kimi_and_every_call_follows_it(monkeypatch):
    """Not an Opus-only stage: configure_route moves every call, including the
    refusal fallback, onto the registry-allowed model chosen for the run."""
    module = _load_editorial_blueprint_module()
    seen = []

    def fake_complete(task_id, **kwargs):
        seen.append((task_id, kwargs["provider"], kwargs["model"]))
        return _reply('[{"title": "T"}]')

    monkeypatch.setattr(module, "complete", fake_complete)
    previous = (module.PROVIDER, module.MODEL_ID)
    try:
        assert module.configure_route(model="openrouter/moonshotai/kimi-k3") == (
            "openrouter", "openrouter/moonshotai/kimi-k3",
        )
        module._call("prompt")
        assert seen == [(module.TASK_ID, "openrouter", "openrouter/moonshotai/kimi-k3")]
        assert module.MODEL_ID == "openrouter/moonshotai/kimi-k3"
    finally:
        module.PROVIDER, module.MODEL_ID = previous


def test_section_slides_accept_wrapped_and_single_slide_replies(monkeypatch):
    """A section whose reply is not a bare list must not silently author 0 slides."""
    module = _load_editorial_blueprint_module()
    slide = {"title": "T", "narration": "N", "mentions": []}
    replies = [
        {"slides": [slide]},
        {"section": "Funnel", "content_slides": [slide, slide]},
        slide,
        [slide, "junk"],
        {"note": "nothing usable"},
    ]

    monkeypatch.setattr(module, "_call", lambda prompt, *, max_tokens=8000: replies.pop(0))
    kwargs = dict(
        topic_title="Demo",
        section={"title": "S", "intent": "I", "slide_budget": 2},
        section_sources=[],
        evidence="E",
        prior_titles=[],
    )
    counts = [len(module.author_section_slides(**kwargs)) for _ in range(5)]
    assert counts == [1, 2, 1, 1, 0]


def test_extra_slides_are_capped_and_never_raise(monkeypatch):
    module = _load_editorial_blueprint_module()
    slide = {"title": "T", "narration": "N", "mentions": []}
    sources = [{"source_id": "paper_abstract:x"}]
    kwargs = dict(
        topic_title="Demo",
        section={"title": "S", "intent": "I", "slide_budget": 5},
        section_sources=sources,
        evidence="E",
        prior_titles=[],
    )
    seen_budgets = []

    def fake_call(prompt, *, max_tokens=8000):
        seen_budgets.append(prompt)
        return [slide] * 4

    monkeypatch.setattr(module, "_call", fake_call)
    distinct = [dict(slide, title=f"T{i}") for i in range(4)]
    monkeypatch.setattr(module, "_call", lambda prompt, *, max_tokens=8000: distinct)
    assert [s["title"] for s in module._author_extra_slides(count=2, **kwargs)] == ["T0", "T1"]
    # Titles already in the deck (any case) are dropped: the assembler would drop them anyway.
    repeated = module._author_extra_slides(count=3, **{**kwargs, "prior_titles": ["t0", "T2"]})
    assert [s["title"] for s in repeated] == ["T1", "T3"]
    monkeypatch.setattr(module, "_call", fake_call)
    assert len(module._author_extra_slides(count=2, **kwargs)) == 1, "identical titles count once"
    assert module._author_extra_slides(count=0, **kwargs) == []
    assert module._author_extra_slides(count=2, **{**kwargs, "section_sources": []}) == []

    def refusing_call(prompt, *, max_tokens=8000):
        raise ValueError("model returned no text (stop_reason='refusal')")

    monkeypatch.setattr(module, "_call", refusing_call)
    assert module._author_extra_slides(count=2, **kwargs) == []


def test_bare_talk_and_paper_titles_never_yield_speakers():
    module = _load_editorial_blueprint_module()
    for title in (
        "FLIP2: Expanding Protein Fitness Landscape Benchmarks for Real-World Machine Learning Applications",
        "Automating Scientific Hypothesis Validation with AI Agents",
    ):
        assert module._people_from_title(title) == [], title
    assert module._people_from_title("Alex Hanna, 'The Uncontroversial Thingness of Computer Vision', AI for Peace workshop") == ["Alex Hanna"]
    assert module._people_from_title("Lewis Kirsch and Jeff Clune, RSI and Post-AGI Science and Society workshops") == ["Lewis Kirsch", "Jeff Clune"]


def test_event_titles_never_yield_speakers():
    """Workshop/expo titles name fields, not people (ICML 2026: 'Machine Learning' as a speaker)."""
    module = _load_editorial_blueprint_module()
    for title in (
        "Workshop: AI as a Tool for Mathematics, Computer Science, and Machine Learning",
        "Expo Workshop AI for Science: Foundation Models and Agentic Systems for Closed-Loop Discovery",
        "3rd AI for Math Workshop: Toward Self-Evolving Scientific Agents",
        "Tutorial: Structured Probabilistic Inference and Generative Modeling",
    ):
        assert module._people_from_title(title) == [], title


def test_blueprint_markdown_shows_each_slide_in_full():
    module = _load_editorial_blueprint_module()
    blueprint = {
        "audience_title": "Demo talk",
        "telescopic_section_map": [{"section_id": "S1", "title": "The loop"}],
        "slides": [
            {
                "section_id": "S1",
                "title": "Loops need checkers",
                "intent": "Why exact checks matter",
                "visible_text": ["Propose", "Test"],
                "claims": ["Exact checkers make loops reliable."],
                "narration_beats": ["Alice showed X; Bob showed Y."],
                "visual_figure_brief": {"depiction": "pipeline", "summary": "Propose to test", "components": ["Proposer", "Checker"]},
                "source_plan": [{"label": "SRC-001", "role": "grounds the claim", "supports": "Alice's X result."}],
                "speaker_author_work_mentions": [
                    {"label": "SRC-001", "source_id": "paper_abstract:a", "title": "Paper A", "authors": ["Alice"], "contribution_note": "Shows X."},
                    {"label": "SRC-002", "source_id": "speech_transcript:b", "title": "Talk B", "speakers": [], "contribution_note": "Shows Y."},
                ],
            }
        ],
    }
    md = module.render_blueprint_markdown(blueprint, topic_title="Demo", stats={"slides": 1, "sources": 2, "works": 2, "people": 1})
    for expected in (
        "**The loop**", "1. Loops need checkers (2 sources)", "**Section goal.** Why exact checks matter",
        "- Propose", "Exact checkers make loops reliable.", "> Alice showed X", "**Figure** (pipeline)",
        "SRC-001 (paper) *Paper A* — Alice — role: grounds the claim", "  - Alice's X result.",
        "  - Shows Y.",
        "SRC-002 (talk) *Talk B* — no named people",
    ):
        assert expected in md, expected
