"""The two figure-compiler CLIs, proven without spending a cent.

``generate_figure_specs.py --dry-run`` writes prompts and placeholder specs;
``compile_figures.py`` turns specs into SVGs, manifests and a receipt. Together
they are the whole free half of the figure path, so both are exercised here
end to end on a synthetic deck.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    path = REPO_ROOT / "scripts" / "presentations" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HAWK_NOTE = (
    "The 'Hawk' method prunes 80.2% of visual tokens in multimodal LLMs, maintaining 96.0% "
    "of original performance while achieving a 1.34x end-to-end speedup."
)


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    deck_dir = tmp_path / "decks" / "01_topic"
    deck_dir.mkdir(parents=True)
    (deck_dir / "presentation_deck.json").write_text(
        json.dumps({
            "presentation_id": "topic-01",
            "slides": [
                {"slide_id": "sl_01_intro", "display_title": "Where this field is going",
                 "visual_role": "mechanism_pipeline",
                 "narration_beats": ["Three threads carry this year's work."],
                 "required_entities": ["HAWK"]},
                # It cites the source its numbers come from -- which is what
                # entitles it to plot them.
                {"slide_id": "sl_02_pruning", "display_title": "Pruning visual tokens",
                 "visual_role": "evidence_chart", "source_ids": ["SRC-014"],
                 "narration_beats": [HAWK_NOTE]},
                {"slide_id": "sl_zz_events", "slide_kind": "events", "display_title": "Key events"},
            ],
        }), encoding="utf-8",
    )
    (deck_dir / "speaker_author_work_ledger.json").write_text(
        json.dumps({"entries": [{"source_id": "SRC-014", "contribution_note": HAWK_NOTE}]}),
        encoding="utf-8",
    )
    return deck_dir


def test_dry_run_authors_a_spec_per_editorial_slide_without_a_provider_call(deck: Path) -> None:
    module = _load("generate_figure_specs.py")
    receipt = module.generate(deck_dir=deck, workspace=deck.parent.parent, dry_run=True)

    assert receipt["status"] == "pass"
    assert receipt["slide_count"] == 2, "the structural bookend slide must not get a figure"
    assert receipt["spec_count"] == 2
    assert receipt["model"] == "dry-run"
    assert receipt["numeric_ledger_claims"] == 3  # 80.2%, 96.0%, 1.34x

    specs = sorted((deck / "figure_specs").glob("*.json"))
    assert [path.name for path in specs] == [
        "sl_01_intro-figure.json", "sl_02_pruning-figure.json",
    ]
    prompts = sorted((deck / "figure_specs").glob("*.prompt.txt"))
    assert len(prompts) == 2
    prompt = prompts[1].read_text(encoding="utf-8")
    assert "80.2" in prompt, "the prompt must offer the slide's measured rows"
    assert "Never invent illustrative values" in prompt
    assert "You do NOT decide how it looks" in prompt


def test_the_slide_with_measured_rows_is_offered_them_and_the_other_is_not(deck: Path) -> None:
    module = _load("generate_figure_specs.py")
    receipt = module.generate(deck_dir=deck, workspace=deck.parent.parent, dry_run=True)
    by_slide = {row["slide_id"]: row for row in receipt["specs"]}
    assert by_slide["sl_02_pruning"]["ledger_rows_offered"]
    assert by_slide["sl_01_intro"]["ledger_rows_offered"] == []
    # With no measured row, the evidence chart is demoted rather than faked.
    assert by_slide["sl_02_pruning"]["selection"]["proposed_archetype"] == "bar_compare"


def test_compile_writes_svgs_manifests_and_a_receipt(deck: Path) -> None:
    _load("generate_figure_specs.py").generate(
        deck_dir=deck, workspace=deck.parent.parent, dry_run=True
    )
    compile_module = _load("compile_figures.py")
    receipt = compile_module.compile_deck(deck, vision_deck=False, allow_variety_findings=True, allow_placeholders=True)

    assert receipt["status"] == "pass"
    assert receipt["compiled_count"] == 2
    assert receipt["blockers"] == []
    for row in receipt["figures"]:
        svg = deck / row["svg_path"]
        manifest = deck / row["manifest_path"]
        assert svg.is_file() and manifest.is_file()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["archetype"] == row["archetype"]
        assert payload["spec_hash"] == row["spec_hash"]
        assert "<svg" in svg.read_text(encoding="utf-8")
    assert receipt["variety"]["total_figures"] == 2


def test_compile_uses_active_authoring_receipt_instead_of_stale_specs(deck: Path) -> None:
    authored = _load("generate_figure_specs.py").generate(
        deck_dir=deck, workspace=deck.parent.parent, dry_run=True
    )
    (deck / "figure_spec_receipt.json").write_text(json.dumps(authored), encoding="utf-8")
    (deck / "figure_specs" / "obsolete-figure.json").write_text("{}", encoding="utf-8")

    receipt = _load("compile_figures.py").compile_deck(
        deck, allow_variety_findings=True, allow_placeholders=True
    )

    assert receipt["status"] == "pass"
    assert receipt["compiled_count"] == 2
    assert {row["figure_id"] for row in receipt["figures"]} == {
        "sl_01_intro-figure", "sl_02_pruning-figure"
    }


def test_compile_blocks_missing_active_spec(deck: Path) -> None:
    authored = _load("generate_figure_specs.py").generate(
        deck_dir=deck, workspace=deck.parent.parent, dry_run=True
    )
    (deck / "figure_spec_receipt.json").write_text(json.dumps(authored), encoding="utf-8")
    (deck / authored["specs"][0]["spec_path"]).unlink()

    receipt = _load("compile_figures.py").compile_deck(
        deck, allow_variety_findings=True, allow_placeholders=True
    )

    assert receipt["status"] == "blocked"
    assert "figure_spec_receipt_file_missing" in {row["code"] for row in receipt["blockers"]}


def test_compile_refuses_a_partial_failed_authoring_receipt(deck: Path) -> None:
    authored = _load("generate_figure_specs.py").generate(
        deck_dir=deck, workspace=deck.parent.parent, dry_run=True
    )
    authored["status"] = "blocked"
    authored["specs"].pop()
    (deck / "figure_spec_receipt.json").write_text(json.dumps(authored), encoding="utf-8")

    receipt = _load("compile_figures.py").compile_deck(
        deck, allow_variety_findings=True, allow_placeholders=True
    )

    assert receipt["status"] == "blocked"
    codes = {row["code"] for row in receipt["blockers"]}
    assert {"figure_spec_authoring_not_pass", "figure_spec_receipt_incomplete"} <= codes


def test_targeted_authoring_preserves_full_active_spec_receipt(deck: Path) -> None:
    module = _load("generate_figure_specs.py")
    prior = module.generate(deck_dir=deck, workspace=deck.parent.parent, dry_run=True)
    update = dict(prior)
    update["specs"] = [{**prior["specs"][0], "archetype": "flow"}]
    update["slide_count"] = update["spec_count"] = 1

    merged = module.merge_targeted_receipt(deck, prior, update)

    assert merged["status"] == "pass"
    assert merged["slide_count"] == merged["spec_count"] == 2
    assert [row["slide_id"] for row in merged["specs"]] == [
        "sl_01_intro", "sl_02_pruning"
    ]
    assert merged["specs"][0]["archetype"] == "flow"
    assert merged["specs"][1] == prior["specs"][1]
    incomplete = module.merge_targeted_receipt(deck, {}, update)
    assert incomplete["status"] == "blocked"
    assert incomplete["blockers"][0]["code"] == "figure_spec_receipt_incomplete_after_targeted_update"


def test_a_spec_citing_a_row_the_deck_ledger_lacks_is_blocked_before_drawing(deck: Path) -> None:
    _load("generate_figure_specs.py").generate(
        deck_dir=deck, workspace=deck.parent.parent, dry_run=True
    )
    spec_path = deck / "figure_specs" / "sl_01_intro-figure.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["grounding"] = {"kind": "data", "ledger_rows": ["nc-invented-0001"]}
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    receipt = _load("compile_figures.py").compile_deck(deck)
    codes = [blocker["code"] for blocker in receipt["blockers"]]
    assert "figure_spec_ledger_row_missing" in codes
    assert receipt["status"] == "blocked"


def test_a_deck_of_one_family_blocks_unless_the_operator_allows_it(deck: Path) -> None:
    """Monotony is a build failure, not a note in a receipt nobody opens: the
    canary shipped 27 box-and-arrow slides before a reviewer said so."""

    _load("generate_figure_specs.py").generate(
        deck_dir=deck, workspace=deck.parent.parent, dry_run=True
    )
    compile_module = _load("compile_figures.py")

    blocked = compile_module.compile_deck(deck, vision_deck=True, allow_placeholders=True)
    assert blocked["status"] == "blocked"
    assert [row["code"] for row in blocked["blockers"]] == ["figure_deck_variety_quota"]
    assert "visual-domain" in blocked["blockers"][0]["detail"]

    allowed = compile_module.compile_deck(deck, vision_deck=True, allow_variety_findings=True,
                                          allow_placeholders=True)
    assert allowed["status"] == "pass"
    assert allowed["variety"]["findings"], "the mix is still reported, just not blocking"


def test_the_gallery_compiles_one_example_of_every_archetype(tmp_path: Path) -> None:
    receipt = _load("compile_figures.py").build_gallery(tmp_path / "gallery")
    assert receipt["status"] == "pass"
    assert receipt["failures"] == []
    assert len(receipt["compiled"]) == receipt["archetypes"] >= 35


# -- the spec-hash cache (design spec §7: "Cache key = spec hash") -------------

def _compiled(deck: Path):
    _load("generate_figure_specs.py").generate(
        deck_dir=deck, workspace=deck.parent.parent, dry_run=True
    )
    return _load("compile_figures.py")


def test_an_unchanged_spec_is_not_recompiled(deck: Path) -> None:
    """The same spec and compiler produce byte-identical SVG, so recompiling it
    can only reproduce the same bytes. The second run should reuse them."""

    module = _compiled(deck)
    first = module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)
    assert first["cache_hit_count"] == 0
    assert all(row["cached"] is False for row in first["figures"])

    second = module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)
    assert second["cache_hit_count"] == first["compiled_count"]
    assert all(row["cached"] is True for row in second["figures"])
    assert [row["svg_sha256"] for row in second["figures"]] == [
        row["svg_sha256"] for row in first["figures"]
    ]


def test_editing_one_spec_recompiles_only_that_figure(deck: Path) -> None:
    module = _compiled(deck)
    module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)

    spec_path = deck / "figure_specs" / "sl_01_intro-figure.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["takeaway"] = "A different takeaway"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    receipt = module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)
    by_id = {row["figure_id"]: row for row in receipt["figures"]}
    assert by_id["sl_01_intro-figure"]["cached"] is False
    assert by_id["sl_02_pruning-figure"]["cached"] is True


def test_a_tampered_svg_is_recompiled_rather_than_trusted(deck: Path) -> None:
    """The cache verifies the file on disk still hashes to what it recorded, so
    a truncated or hand-edited SVG is rebuilt instead of being served."""

    module = _compiled(deck)
    module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)
    svg = sorted((deck / "assets" / "compiled").glob("*.svg"))[0]
    svg.write_text(svg.read_text(encoding="utf-8") + "<!-- edited by hand -->", encoding="utf-8")

    receipt = module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)
    cached = {row["figure_id"]: row["cached"] for row in receipt["figures"]}
    assert sum(1 for value in cached.values() if value is False) == 1
    assert svg.read_text(encoding="utf-8").endswith("</svg>\n"), "the tampered file was rebuilt"


def test_no_cache_recompiles_everything(deck: Path) -> None:
    module = _compiled(deck)
    module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)
    receipt = module.compile_deck(deck, allow_variety_findings=True, use_cache=False, allow_placeholders=True)
    assert receipt["cache_hit_count"] == 0


def test_a_cached_figure_still_reports_a_missing_required_entity(deck: Path) -> None:
    """A cache hit must not launder a blocker the first run would have raised."""

    module = _compiled(deck)
    spec_path = deck / "figure_specs" / "sl_01_intro-figure.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["required_entities"] = ["A Phrase That Is Nowhere In The Figure"]
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    first = module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)
    second = module.compile_deck(deck, allow_variety_findings=True, allow_placeholders=True)
    codes = lambda receipt: sorted(row["code"] for row in receipt["blockers"])  # noqa: E731
    assert "figure_required_entity_missing" in codes(first)
    assert codes(second) == codes(first), "the cached run must raise the same blocker"


def test_the_model_response_is_unwrapped_the_way_the_receipt_boundary_returns_it(monkeypatch) -> None:
    """The paid path's response unwrapping, which --dry-run never exercises.

    ``anthropic_messages_create`` returns a ProviderCallResult that delegates
    attribute access to the SDK response. Reading ``.value`` off it raised
    AttributeError on the first real call and cost a live request to discover.
    The unwrapping now lives in the shared client's Anthropic adapter, which
    ``_call_model`` goes through for every provider.
    """

    from conference_pipeline.model_governance import default_model, llm_client

    class _Block:
        type = "text"

        def __init__(self, text: str) -> None:
            self.text = text

    class _Response:
        stop_reason = "end_turn"
        content = [_Block("hello "), _Block("world")]

    class _Result:  # mirrors ProviderCallResult's delegation
        def __init__(self, response: object) -> None:
            self._response = response

        def __getattr__(self, name: str):
            return getattr(self._response, name)

    module = _load("generate_figure_specs.py")
    captured: dict[str, object] = {}

    def _fake_create(client, **kwargs):
        captured.update(kwargs)
        return _Result(_Response())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "anthropic_messages_create", _fake_create)
    monkeypatch.setattr(llm_client, "_anthropic_client", lambda request: object())

    model = default_model("presentation.figure_spec_authoring")
    text = module._call_model("a prompt", model=model, caller=__file__)
    assert text == "hello world"
    assert captured["model"] == model
    assert captured["thinking"] == {"type": "disabled"}, "authoring never spends its budget thinking"


def test_dict_shaped_content_blocks_are_also_unwrapped(monkeypatch) -> None:
    from conference_pipeline.model_governance import default_model, llm_client

    class _Result:
        stop_reason = "end_turn"
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

    module = _load("generate_figure_specs.py")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "anthropic_messages_create", lambda client, **kw: _Result())
    monkeypatch.setattr(llm_client, "_anthropic_client", lambda request: object())
    model = default_model("presentation.figure_spec_authoring")
    assert module._call_model("p", model=model, caller=__file__) == "ab"


def test_figure_specs_can_be_authored_by_kimi_through_openrouter(monkeypatch) -> None:
    """The authoring stage is not an Anthropic stage: a model the registry allows
    for the task is routed to its own provider, with thinking switched off."""

    from types import SimpleNamespace

    from conference_pipeline.model_governance import llm_client

    module = _load("generate_figure_specs.py")
    sent: dict[str, object] = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            id="g", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'), finish_reason="stop")],
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "_chat_client", lambda provider, request: fake)
    text = module._call_model(
        "a prompt", model="openrouter/moonshotai/kimi-k3", caller=__file__,
        task_id="presentation.figure_spec_critic",
    )
    assert text == '{"ok": true}'
    assert sent["model"] == "moonshotai/kimi-k3"
    assert sent["extra_body"] == {"reasoning": {"enabled": False}}


def test_a_dry_run_placeholder_never_satisfies_a_resume(deck: Path) -> None:
    """A placeholder validates cleanly, which is exactly why it is dangerous.

    A --dry-run preflight leaves valid specs on disk. If `skip_existing` accepts
    them, the preflight silently becomes the deck's real content and every gate
    downstream agrees it is fine.
    """

    module = _load("generate_figure_specs.py")
    module.generate(deck_dir=deck, workspace=deck.parent.parent, dry_run=True)
    specs = sorted((deck / "figure_specs").glob("*.json"))
    assert specs
    payload = json.loads(specs[0].read_text(encoding="utf-8"))
    assert module.PLACEHOLDER_NOTE in payload["grounding"]["note"]

    calls: list[str] = []
    module._call_model = lambda *a, **k: calls.append("called") or "{}"
    import os
    previous = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
    try:
        module.generate(deck_dir=deck, workspace=deck.parent.parent,
                        dry_run=False, skip_existing=True)
    except Exception:
        pass
    finally:
        if previous is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = previous
    assert calls, "a placeholder must send the slide back to the model, not be reused"


def test_resume_reauthors_a_structurally_valid_spec_the_compiler_refuses(deck: Path) -> None:
    module = _load("generate_figure_specs.py")
    module.generate(deck_dir=deck, workspace=deck.parent.parent, dry_run=True,
                    slide_ids=("sl_01_intro",))
    path = deck / "figure_specs" / "sl_01_intro-figure.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["grounding"].pop("note", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    from conference_pipeline.figures.errors import FigureLayoutError

    def refuse(_payload):
        raise FigureLayoutError("no room for the return edge")

    module.compile_figure = refuse
    calls: list[str] = []
    module._call_model = lambda *a, **k: calls.append("called") or "{}"
    import os
    previous = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
    try:
        module.generate(deck_dir=deck, workspace=deck.parent.parent,
                        dry_run=False, skip_existing=True, slide_ids=("sl_01_intro",))
    finally:
        if previous is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = previous
    assert calls, "a compiler refusal must re-enter authoring on resume"


def test_resume_reuses_a_compiling_spec_without_repeating_paid_critique(deck: Path) -> None:
    module = _load("generate_figure_specs.py")
    module.generate(deck_dir=deck, workspace=deck.parent.parent, dry_run=True,
                    slide_ids=("sl_01_intro",))
    path = deck / "figure_specs" / "sl_01_intro-figure.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["grounding"].pop("note", None)
    payload["title"] = "Research threads"
    path.write_text(json.dumps(payload), encoding="utf-8")
    module.compile_figure(payload)

    def unexpected(*args, **kwargs):
        raise AssertionError("a passing spec must reuse its authored critique")

    module._call_model = unexpected
    import os
    previous = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
    try:
        receipt = module.generate(
            deck_dir=deck, workspace=deck.parent.parent, dry_run=False,
            skip_existing=True, slide_ids=("sl_01_intro",), critic=True,
            critic_runner=unexpected,
        )
    finally:
        if previous is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = previous
    assert receipt["status"] == "pass"
    assert receipt["critic"]["calls"] == 0


def test_a_reused_spec_keeps_its_recorded_critique_in_the_receipt(deck: Path) -> None:
    """A resumed run reported 0 open findings for ICML deck 02, which had 17:
    reused specs dropped their critique from the receipt."""

    module = _load("generate_figure_specs.py")
    module.generate(deck_dir=deck, workspace=deck.parent.parent, dry_run=True, slide_ids=("sl_01_intro",))
    path = deck / "figure_specs" / "sl_01_intro-figure.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["grounding"].pop("note", None)
    payload["title"] = "Research threads"
    path.write_text(json.dumps(payload), encoding="utf-8")
    recorded = {"verdict": "revise", "critic_calls": 3,
                "unresolved_major": [{"code": "overclaim", "severity": "major", "detail": "d", "suggested_change": "c"}]}
    (deck / "figure_specs" / "sl_01_intro-figure.critic.json").write_text(json.dumps(recorded), encoding="utf-8")

    def unexpected(*args, **kwargs):
        raise AssertionError("a reused spec is not critiqued again")

    module._call_model = unexpected
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    receipt = module.generate(deck_dir=deck, workspace=deck.parent.parent, skip_existing=True,
                              slide_ids=("sl_01_intro",), critic_runner=unexpected)
    assert receipt["critic"]["calls"] == 0
    assert receipt["critic"]["unresolved_major"][0]["issues"][0]["code"] == "overclaim"


def test_resume_reauthors_a_cached_duplicate_heading(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    deck_dir = _critic_deck(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    old = {**_good_flow_spec(module), "title": "From images to video"}
    path = out / "sl_01_fusion-figure.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    repaired = {**old, "title": "Temporal fusion mechanism"}
    calls = []
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: calls.append(a) or json.dumps(repaired))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    receipt = module.generate(deck_dir=deck_dir, workspace=tmp_path, spec_dir=out,
                              skip_existing=True, critic=False)
    assert receipt["status"] == "pass"
    assert len(calls) == 1
    assert json.loads(path.read_text())["title"] == "Temporal fusion mechanism"

    def unexpected(*args, **kwargs):
        raise AssertionError("a distinct cached heading needs no provider call")

    monkeypatch.setattr(module, "_call_model", unexpected)
    assert module.generate(deck_dir=deck_dir, workspace=tmp_path, spec_dir=out,
                           skip_existing=True, critic=False)["status"] == "pass"


def test_compilable_duplicate_heading_cannot_use_budget_fallback(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    repeated = {**_good_flow_spec(module), "title": "From images to video"}
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: json.dumps(repeated))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    receipt = module.generate(deck_dir=_critic_deck(tmp_path), workspace=tmp_path,
                              spec_dir=tmp_path / "out", critic=False)
    assert receipt["status"] == "blocked"
    assert "figure_title_repeats_slide_title" in receipt["blockers"][0]["errors"][0]


def test_critic_revision_cannot_reintroduce_duplicate_heading(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    good = {**_good_flow_spec(module), "title": "Temporal fusion mechanism"}
    repeated = {**good, "title": "From images to video"}
    responses = iter([good, repeated, good])
    prompts = []

    def fake_call(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(next(responses))

    monkeypatch.setattr(module, "_call_model", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    verdicts = iter([_major("overclaim"), {"verdict": "accept", "issues": []}])
    out = tmp_path / "out"
    receipt = module.generate(deck_dir=_critic_deck(tmp_path), workspace=tmp_path,
                              spec_dir=out, critic_runner=lambda *a: next(verdicts))
    assert receipt["status"] == "pass"
    assert "figure_title_repeats_slide_title" in prompts[-1]
    assert json.loads((out / "sl_01_fusion-figure.json").read_text())["title"] == good["title"]


def test_compiler_repair_cannot_reintroduce_duplicate_heading(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    good = {**_good_flow_spec(module), "title": "Temporal fusion mechanism"}
    repeated = {**good, "title": "From images to video"}
    responses = iter([good, repeated])
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: json.dumps(next(responses)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def refuses(spec):
        raise module.FigureCompilerError("label does not fit")

    monkeypatch.setattr(module, "compile_figure", refuses)
    out = tmp_path / "out"
    receipt = module.generate(deck_dir=_critic_deck(tmp_path), workspace=tmp_path,
                              spec_dir=out, critic=False)
    assert receipt["status"] == "blocked"
    assert json.loads((out / "sl_01_fusion-figure.json").read_text())["title"] == good["title"]


def test_compiling_a_placeholder_spec_is_blocked(deck: Path) -> None:
    _load("generate_figure_specs.py").generate(
        deck_dir=deck, workspace=deck.parent.parent, dry_run=True
    )
    receipt = _load("compile_figures.py").compile_deck(deck, allow_variety_findings=True)
    codes = {row["code"] for row in receipt["blockers"]}
    assert "figure_spec_is_a_dry_run_placeholder" in codes
    assert receipt["status"] == "blocked"


def test_authoring_can_target_named_slides_and_a_scratch_directory(tmp_path: Path) -> None:
    """Re-authoring ten slides should not cost a run of forty-nine.

    And a dry run must be able to write its placeholders somewhere other than
    the real spec directory, which it would otherwise clobber.
    """

    module = _load("generate_figure_specs.py")
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "presentation_deck.json").write_text(
        json.dumps({
            "slides": [
                {"slide_id": "sl_01_first", "title": "First", "visual_role": "mechanism_pipeline"},
                {"slide_id": "sl_02_second", "title": "Second", "visual_role": "mechanism_pipeline"},
                {"slide_id": "sl_03_third", "title": "Third", "visual_role": "mechanism_pipeline"},
            ]
        }),
        encoding="utf-8",
    )
    out = tmp_path / "scratch"
    receipt = module.generate(
        deck_dir=deck, workspace=tmp_path, dry_run=True,
        slide_ids=("sl_02",), spec_dir=out,
    )
    assert receipt["spec_count"] == 1
    written = sorted(path.name for path in out.glob("*.json"))
    assert any("sl_02" in name for name in written), written
    assert not (deck / "figure_specs").exists(), "a dry run wrote into the real spec directory"


def test_the_prompt_carries_the_slide_contract_the_deck_actually_has(tmp_path: Path) -> None:
    """It was reading four keys that do not exist on this contract.

    The claim line rendered EMPTY on every slide of both canary decks, and the
    blueprint's components and edges -- which ARE the figure's content -- were
    handed over buried inside one 900-character string.
    """

    module = _load("generate_figure_specs.py")
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "presentation_deck.json").write_text(
        json.dumps({"slides": [{
            "slide_id": "sl_01_fusion",
            "title": "From images to video",
            "section_id": "S1",
            "claims": ["Extending VLMs to video strains simple fusion."],
            "visible_text": ["Video adds time: motion, tracking, persistence"],
            "visual_role": (
                "mechanism_pipeline: temporal fusion feeding a framework "
                "| components: Single-image fusion; Temporal fusion "
                "| edges: Single-image fusion -> Temporal fusion"
            ),
        }]}),
        encoding="utf-8",
    )
    out = tmp_path / "scratch"
    module.generate(deck_dir=deck, workspace=tmp_path, dry_run=True, spec_dir=out)
    prompt = next(out.glob("*.prompt.txt")).read_text(encoding="utf-8")

    assert "Extending VLMs to video strains simple fusion." in prompt
    assert "Video adds time: motion, tracking, persistence" in prompt
    assert "- Single-image fusion -> Temporal fusion" in prompt
    assert "STRUCTURE RULES" in prompt
    assert "A filmstrip's frames are moments of one sequence" in prompt
    assert "content keys this archetype reads" in prompt


def test_the_prompt_states_what_omitting_a_glyph_draws(tmp_path: Path) -> None:
    """A panel with no glyph gets a landscape thumbnail, and nobody was told."""

    module = _load("generate_figure_specs.py")
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "presentation_deck.json").write_text(
        json.dumps({"slides": [{
            "slide_id": "sl_01_flicker",
            "title": "Where per-frame editing flickers",
            "visual_role": "mechanism_pipeline: why per-frame editing flickers between frames",
        }]}),
        encoding="utf-8",
    )
    out = tmp_path / "scratch"
    module.generate(deck_dir=deck, workspace=tmp_path, dry_run=True, spec_dir=out)
    prompt = next(out.glob("*.prompt.txt")).read_text(encoding="utf-8")
    assert "draws a photo" in prompt and "generic landscape" in prompt


def _critic_deck(tmp_path: Path) -> Path:
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "presentation_deck.json").write_text(
        json.dumps({"slides": [{
            "slide_id": "sl_01_fusion",
            "title": "From images to video",
            "claims": ["Extending VLMs to video strains simple fusion."],
            "visual_role": "mechanism_pipeline: temporal fusion feeding a framework",
        }]}),
        encoding="utf-8",
    )
    return deck


def test_the_critic_is_on_by_default(tmp_path: Path) -> None:
    """Always on; a dry run simply makes no provider call."""

    module = _load("generate_figure_specs.py")
    receipt = module.generate(
        deck_dir=_critic_deck(tmp_path), workspace=tmp_path, dry_run=True,
        spec_dir=tmp_path / "out",
    )
    assert receipt["critic"]["enabled"] is True
    assert receipt["critic"]["calls"] == 0
    assert receipt["critic"]["max_rounds"] == 3


def test_a_revise_verdict_is_recorded_beside_the_spec(tmp_path: Path) -> None:
    module = _load("generate_figure_specs.py")
    out = tmp_path / "out"

    def runner(payload, slide, archetype):
        return {
            "verdict": "revise",
            "issues": [{"code": "alternatives_as_sequence",
                        "detail": "a reader reads the second outcome as following the first",
                        "suggested_change": "use branching_paths"}],
            "rubric_version": module.CRITIC_RUBRIC_VERSION,
            "model_task_id": module.CRITIC_TASK_ID,
        }

    receipt = module.generate(
        deck_dir=_critic_deck(tmp_path), workspace=tmp_path, dry_run=True,
        spec_dir=out, critic=True, critic_runner=runner,
    )
    assert receipt["critic"]["calls"] == 1
    assert receipt["critic"]["verdicts"]["revise"] == 1
    sidecar = json.loads(next(out.glob("*.critic.json")).read_text(encoding="utf-8"))
    assert sidecar["issues"][0]["code"] == "alternatives_as_sequence"


def test_the_critic_halts_before_spending_when_its_worst_case_exceeds_the_cap(
    tmp_path: Path, monkeypatch
) -> None:
    """A cap that halts is the repo's rule; the critic never silently skips slides."""

    module = _load("generate_figure_specs.py")
    deck = _critic_deck(tmp_path)
    contract = json.loads((deck / "presentation_deck.json").read_text(encoding="utf-8"))
    contract["slides"] = [
        {**contract["slides"][0], "slide_id": f"sl_{i:02d}_slide"} for i in range(6)
    ]
    (deck / "presentation_deck.json").write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def no_spend(*args, **kwargs):
        raise AssertionError("a provider call was made before the budget check")

    monkeypatch.setattr(module, "_call_model", no_spend)
    with pytest.raises(module.CriticBudgetExceeded, match="exceeds --critic-budget-usd"):
        module.generate(
            deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "out",
            critic_budget_usd=module.CRITIC_EST_USD_PER_CALL * 3 * 5,
        )


def _good_flow_spec(module) -> dict:
    return {
        "spec_version": module.SPEC_VERSION, "figure_id": "f", "slide_id": "s",
        "archetype": "flow", "claim": "c", "takeaway": "t",
        "grounding": {"kind": "schematic"},
        "content": {"nodes": [{"id": "a", "label": "A", "kind": "process"},
                              {"id": "b", "label": "B", "kind": "process"}],
                    "edges": [{"from": "a", "to": "b"}]},
    }


def _major(code: str = "unsupported_number") -> dict:
    return {"verdict": "revise",
            "issues": [{"code": code, "severity": "major",
                        "detail": "the 12% is in no numeric row", "suggested_change": "drop it"}]}


def test_major_issues_are_fixed_over_at_most_three_rounds(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: json.dumps(_good_flow_spec(module)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    verdicts = [_major(), _major("misattribution"), {"verdict": "accept", "issues": []}]

    receipt = module.generate(
        deck_dir=_critic_deck(tmp_path), workspace=tmp_path, spec_dir=tmp_path / "out",
        critic_runner=lambda payload, slide, archetype: verdicts.pop(0),
    )
    row = receipt["specs"][0]["critic"]
    assert row["critic_calls"] == 3 and row["reauthored"] is True
    assert row["verdict"] == "accept" and row["unresolved_major"] == []
    assert receipt["critic"]["calls"] == 3 and receipt["critic"]["unresolved_major"] == []


def test_a_major_issue_left_after_three_rounds_is_reported_not_looped(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: json.dumps(_good_flow_spec(module)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def always_major(payload, slide, archetype):
        calls.append(1)
        return _major("overclaim")

    receipt = module.generate(
        deck_dir=_critic_deck(tmp_path), workspace=tmp_path, spec_dir=tmp_path / "out",
        critic_runner=always_major,
    )
    assert len(calls) == 3
    unresolved = receipt["critic"]["unresolved_major"]
    assert unresolved and unresolved[0]["issues"][0]["code"] == "overclaim"


def test_minor_issues_never_buy_a_reauthor(monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    monkeypatch.setattr(
        module, "_call_model",
        lambda *a, **k: json.dumps({"verdict": "revise", "issues": [
            {"code": "emphasis_off_claim", "severity": "minor", "detail": "d", "suggested_change": "c"}]}),
    )
    critique = module._critique({}, {}, "flow", model="m")
    assert critique["verdict"] == "accept" and critique["issues"][0]["severity"] == "minor"


def test_the_critic_prompt_is_adversarial_and_carries_the_evidence() -> None:
    module = _load("generate_figure_specs.py")
    slide = {"title": "T", "claims": ["c"], "named_paper_talk_speaker_author_mentions": [
        {"label": "SRC-001", "title": "Paper A", "authors": ["Alice Smith"]}]}
    evidence = module.critic_evidence(slide, [])
    for needle in ('SOURCE BLOCK', '"label": "SRC-001"', '"title": "Paper A"', '"authors": ["Alice Smith"]'):
        assert needle in evidence
    prompt = module.build_critic_prompt(payload={}, slide=slide, archetype="flow", evidence=evidence)
    for needle in ("ADVERSARIAL", "unsupported_number", "misattribution", "overclaim", "Alice Smith", '"severity"'):
        assert needle in prompt, needle


def test_an_unknown_critic_code_normalises_rather_than_crashes(monkeypatch) -> None:
    """A code outside the rubric becomes "other" instead of poisoning the tally."""

    module = _load("generate_figure_specs.py")
    monkeypatch.setattr(
        module, "_call_model",
        lambda *a, **k: json.dumps({
            "verdict": "revise",
            "issues": [{"code": "vibes", "detail": "d", "suggested_change": "c"}],
        }),
    )
    critique = module._critique({}, {}, "flow", model="m")
    assert critique["issues"][0]["code"] == "other"
    assert critique["verdict"] == "revise"


def test_a_revise_verdict_with_no_issues_is_an_accept(monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    monkeypatch.setattr(
        module, "_call_model", lambda *a, **k: json.dumps({"verdict": "revise", "issues": []})
    )
    assert module._critique({}, {}, "flow", model="m")["verdict"] == "accept"


def test_a_critic_failure_never_stops_authoring(tmp_path: Path) -> None:
    module = _load("generate_figure_specs.py")

    def runner(payload, slide, archetype):
        raise RuntimeError("provider down")

    receipt = module.generate(
        deck_dir=_critic_deck(tmp_path), workspace=tmp_path, dry_run=True,
        spec_dir=tmp_path / "out", critic=True, critic_runner=runner,
    )
    assert receipt["spec_count"] == 1
    assert "provider down" in receipt["specs"][0]["critic"]["error"]


def test_a_revision_that_will_not_compile_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """A critique that trades a compiling figure for a refusing one is not a fix.

    Three of the first fourteen live revisions were structurally valid and
    geometrically impossible, and one named `grid` as a list where the layout
    reads an object -- which reached the layout and raised an AttributeError
    rather than a message anyone could act on.
    """

    module = _load("generate_figure_specs.py")
    deck = _critic_deck(tmp_path)
    calls = {"n": 0}
    good = {
        "spec_version": module.SPEC_VERSION, "figure_id": "f", "slide_id": "s",
        "archetype": "flow", "claim": "c", "takeaway": "t",
        "grounding": {"kind": "schematic"},
        "content": {"nodes": [{"id": "a", "label": "A", "kind": "process"},
                              {"id": "b", "label": "B", "kind": "process"}],
                    "edges": [{"from": "a", "to": "b"}]},
    }
    # The first call authors a spec that compiles; the revision names `grid` as
    # a list, which is structurally wrong and used to reach the layout.
    bad = {**good, "archetype": "token_grid_overlay",
           "content": {"grid": [], "notes": []}}

    def fake_call(prompt, **kwargs):
        calls["n"] += 1
        return json.dumps(good if calls["n"] == 1 else bad)

    monkeypatch.setattr(module, "_call_model", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def runner(payload, slide, archetype):
        return {"verdict": "revise",
                "issues": [{"code": "wrong_archetype", "detail": "d", "suggested_change": "c"}]}

    receipt = module.generate(
        deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "out",
        critic=True, critic_runner=runner,
    )
    row = receipt["specs"][0]["critic"]
    assert row["reauthored"] is False
    assert "revision_error" in row or row.get("revision_rejected")


# --- the critic's revision may not quietly shrink the spec ---------------
#
# Every case below is drawn from the first paid critic pass, where a
# revision answered a note about two highlighted groups by deleting nine of
# a slide's fourteen abilities. `validate_spec` and `compile_figure` both
# passed it: a smaller tree is a well-formed tree that fits more easily.


def _critique(*issues: str) -> dict:
    return {"verdict": "revise",
            "issues": [{"code": "x", "detail": text, "suggested_change": ""}
                       for text in issues]}


def _tree(*groups: tuple[str, list[str]]) -> dict:
    return {"label": "root",
            "children": [{"label": name,
                          "children": [{"label": leaf} for leaf in leaves]}
                         for name, leaves in groups]}


def test_unrequested_deletion_is_refused():
    """Nine abilities vanish; the critic had only mentioned highlighting."""

    module = _load("generate_figure_specs.py")
    before = _tree(("Perception", ["Counting", "Localization"]),
                   ("Reasoning", ["Comparison", "Aggregation"]))
    after = _tree(("Perception", ["Counting"]))
    lost = module.unrequested_losses(
        before, after, _critique("Highlighting two groups overstates their importance.")
    )
    assert "Reasoning" in lost
    assert "Comparison" in lost


def test_deletion_the_critic_named_is_allowed():
    """`Remove the counterfactual branch` is a request, not a regression."""

    module = _load("generate_figure_specs.py")
    before = _tree(("Perception", ["Counting", "Localization"]),
                   ("Counterfactual", ["Swap the evidence"]))
    after = _tree(("Perception", ["Counting", "Localization"]))
    lost = module.unrequested_losses(
        before, after,
        _critique("Remove the counterfactual branch; swap the evidence is off-claim."),
    )
    assert lost == []


def test_rewording_a_label_is_not_a_loss():
    """An axis relabelled keeps its content, so nothing is refused."""

    module = _load("generate_figure_specs.py")
    before = {"points": [{"label": "A"}, {"label": "B"}],
              "x_axis": {"high": "needs abundant labelled/paired data"}}
    after = {"points": [{"label": "A"}, {"label": "B"}],
             "x_axis": {"high": "abundant paired data"}}
    assert module.unrequested_losses(before, after, _critique("Relabel the axis.")) == []


def test_a_shrunk_collection_is_caught_even_when_labels_survive():
    """Three quadrant labels became two, and the critic never said so."""

    module = _load("generate_figure_specs.py")
    before = {"quadrant_labels": [{"label": "Cheap but brittle", "x": "low", "y": "low"},
                                  {"label": "Cheap and resilient", "x": "low", "y": "high"},
                                  {"label": "Costly but reliable", "x": "high", "y": "high"}]}
    after = {"quadrant_labels": [{"label": "Cheap but brittle", "x": "low", "y": "low"},
                                 {"label": "Cheap and resilient", "x": "low", "y": "high"}]}
    lost = module.unrequested_losses(before, after, _critique("Move the marker lower."))
    assert lost == ["Costly but reliable"]


# -- a section is grounded by what it CITES --------------------------------------


def test_a_section_is_grounded_only_by_a_cited_plottable_row():
    """Text overlap offers candidates; it does not assert a section's numbers.

    One CVPR row about hierarchical diffusion's memory shared three
    distinctive words with claims in editing, video-time and physics, so three
    unrelated sections were told to chart a number belonging to none of them.
    And half the rows carry a value and a unit with no metric, which cannot
    fill an axis at all.
    """

    module = _load("compile_figures.py")
    from conference_pipeline.figures.ledger import NumericClaim, NumericLedger

    ledger = NumericLedger(topic_id="t", claims=[
        NumericClaim(claim_id="nc-1", source_id="s",
                     span="hierarchical diffusion cuts peak memory by 56%",
                     metric="memory", value=56.0, unit="%"),
        NumericClaim(claim_id="nc-2", source_id="s",
                     span="hierarchical diffusion improves consistency by 33%",
                     metric="", value=33.0, unit="%"),
    ])
    claim = "hierarchical diffusion improves consistency"

    # Cited and plottable: the section owns a number it can draw.
    assert module._section_can_show_a_number(ledger, ["nc-1"], claim)
    # Cited but not plottable: charting it means inventing the axis label.
    assert not module._section_can_show_a_number(ledger, ["nc-2"], claim)
    # Matched by wording alone, cited by nobody: a hint, not a measurement.
    assert not module._section_can_show_a_number(ledger, [], claim)
    assert not module._section_can_show_a_number(ledger, [], "")


def test_a_wrong_archetype_verdict_switches_the_archetype_on_revision(tmp_path: Path, monkeypatch) -> None:
    """The one critic finding the re-author could never act on: the selector had
    fixed the archetype before the model saw the slide."""

    from conference_pipeline.figures.examples import EXAMPLES

    module = _load("generate_figure_specs.py")
    deck = _critic_deck(tmp_path)
    calls = {"n": 0}
    first = {
        "spec_version": module.SPEC_VERSION, "figure_id": "f", "slide_id": "s",
        "archetype": "flow", "claim": "Two outcomes follow one input.", "takeaway": "Two outcomes",
        "grounding": {"kind": "schematic"},
        "content": {"nodes": [{"id": "a", "label": "Input", "kind": "process"},
                              {"id": "b", "label": "Outcome one", "kind": "process"},
                              {"id": "c", "label": "Outcome two", "kind": "process"}],
                    "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]},
    }
    branched = json.loads(json.dumps(EXAMPLES["branching_paths"]))
    branched.update({"figure_id": "f", "slide_id": "s"})
    prompts: list[str] = []

    def fake_call(prompt, **kwargs):
        calls["n"] += 1
        prompts.append(prompt)
        return json.dumps(first if calls["n"] == 1 else branched)

    monkeypatch.setattr(module, "_call_model", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def runner(payload, slide, archetype):
        # The revision fixes it, so the second round accepts.
        if archetype == "branching_paths":
            return {"verdict": "accept", "issues": []}
        return {"verdict": "revise",
                "issues": [{"code": "wrong_archetype",
                            "detail": "two outcomes are drawn as consecutive stages",
                            "suggested_change": "author it as branching_paths"}]}

    receipt = module.generate(
        deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "out",
        critic=True, critic_runner=runner,
    )
    row = receipt["specs"][0]
    assert row["critic"]["reauthored"] is True
    assert row["critic"]["archetype_switched"] == "branching_paths"
    assert row["archetype"] == "branching_paths"
    assert "ARCHETYPE TO USE: branching_paths" in prompts[-1]


def test_a_wrong_archetype_verdict_never_names_a_picture_the_brief_outgrows() -> None:
    module = _load("generate_figure_specs.py")
    critique = {"verdict": "revise",
                "issues": [{"code": "wrong_archetype", "detail": "d",
                            "suggested_change": "draw it as a spatial_3d_schematic"}]}
    heavy = ("mechanism_pipeline: seven stages | components: a; b; c; d; e; f; g | edges: "
             "a -> b; b -> c; c -> d; d -> e; e -> f; f -> g")
    assert module.named_archetype_switch(critique, current="flow", visual_role=heavy) == ""
    light = "mechanism_pipeline: a scene | components: a; b; c | edges: a -> b"
    assert module.named_archetype_switch(critique, current="flow", visual_role=light) == "spatial_3d_schematic"


def test_a_switch_is_honoured_under_any_issue_code_not_only_wrong_archetype() -> None:
    """Two real production specs named the fix and it was never applied.

    A branch-then-merge recovery decision forced into a `loop`'s ring got
    critiqued as `alternatives_as_sequence` (per the rubric's own category
    for exactly this mismatch), naming `decision_tree` in `suggested_change`
    -- and a linear diagnostic chain forced into the same ring got critiqued
    as `claim_not_encoded`, naming `flow`. Neither is coded `wrong_archetype`,
    so the old code == "wrong_archetype" gate never even read the switch,
    silently reauthoring the same wrong shape with the critique pasted in as
    advice it had no structure to act on.
    """

    module = _load("generate_figure_specs.py")
    alternatives_as_sequence = {
        "verdict": "revise",
        "issues": [{
            "code": "alternatives_as_sequence",
            "detail": "the claim describes choosing Retry or Reset, not all three in sequence",
            "suggested_change": "Restructure as a decision_tree rooted at the monitor with a "
                                 "first split on 'failure detected?', then branch Retry vs Reset.",
        }],
    }
    assert module.named_archetype_switch(alternatives_as_sequence, current="loop") == "decision_tree"

    claim_not_encoded = {
        "verdict": "revise",
        "issues": [
            {"code": "orphan_element", "detail": "d", "suggested_change": "wire shift_gap in"},
            {"code": "claim_not_encoded",
             "detail": "the claim is a one-pass pipeline, not a cycle",
             "suggested_change": "Use flow archetype: Training distribution -> Frozen model -> "
                                  "Shifted test stream -> Updated model, as a directed pipeline."},
        ],
    }
    assert module.named_archetype_switch(claim_not_encoded, current="loop") == "flow"


def test_an_archetype_name_used_as_an_ordinary_word_in_detail_is_not_a_switch() -> None:
    """`detail` is freeform prose more likely to use "flow" as English than as
    an instruction -- only `suggested_change` (the concrete fix) is matched."""

    module = _load("generate_figure_specs.py")
    critique = {
        "verdict": "revise",
        "issues": [{
            "code": "depiction_mismatch",
            "detail": "a reader would misread the data flow direction here",
            "suggested_change": "swap the arrowhead ends so the causal direction reads correctly",
        }],
    }
    assert module.named_archetype_switch(critique, current="loop") == ""


def test_an_archetype_name_far_from_any_switch_verb_in_the_same_sentence_is_not_a_switch() -> None:
    """A real critique offered two fixes in one sentence: "Switch to
    layered_stack ... or add a feedback edge ... to show the corrective
    loop." `loop` is 150 characters past "Switch", describing the return
    path `layered_stack` already draws -- not naming an archetype. Plain
    same-sentence or unwindowed matching read it as a switch to `loop` and
    broke a `flow` figure the critic asked to become `layered_stack`."""

    module = _load("generate_figure_specs.py")
    critique = {
        "verdict": "revise",
        "issues": [{
            "code": "wrong_archetype",
            "detail": "d",
            "suggested_change": (
                "Switch to layered_stack (frozen core policy, adapted layers stacked "
                "around/above it) or add a feedback edge from recovery back to policy "
                "to show the corrective loop."
            ),
        }],
    }
    assert module.named_archetype_switch(critique, current="flow") == "layered_stack"


def test_an_archetype_name_used_as_an_edge_kind_example_is_not_a_switch() -> None:
    """"flow" is also the generic edge `kind` every archetype can use, not
    only the `flow` archetype's name. "Add explicit edge kinds (e.g. flow
    from X to Y)" is an edge-kind example with no switch verb anywhere --
    reading it as "switch to flow" broke a `nested_architecture` figure the
    critic never asked to change archetype at all."""

    module = _load("generate_figure_specs.py")
    critique = {
        "verdict": "revise",
        "issues": [{
            "code": "orphan_element",
            "detail": "d",
            "suggested_change": (
                "Add explicit edge kinds (e.g. flow from slow_path to fast_path) and "
                "connect automoma into the same graph so the diagram shows one "
                "integrated pipeline rather than two unlinked clusters."
            ),
        }],
    }
    assert module.named_archetype_switch(critique, current="nested_architecture") == ""


def test_a_compile_refusal_is_fed_back_to_the_author_once(tmp_path: Path, monkeypatch) -> None:
    """A spec that validates and then refuses at compile used to reach the
    operator after the whole deck had run; the author gets one cheap turn first."""

    module = _load("generate_figure_specs.py")
    deck = _critic_deck(tmp_path)
    calls = {"n": 0}
    # A label that reads as a cut-off sentence: no ladder rung rewords text,
    # so the compiler refuses this deterministically (deck 04, sl_01).
    refused = {
        "spec_version": module.SPEC_VERSION, "figure_id": "f", "slide_id": "s",
        "archetype": "flow", "claim": "Agents act on what they perceive.", "takeaway": "Perceive then act",
        "grounding": {"kind": "schematic"},
        "content": {"nodes": [{"id": "a", "label": "what the embodied agents actually are", "kind": "process"},
                              {"id": "b", "label": "Action", "kind": "process"}],
                    "edges": [{"from": "a", "to": "b"}]},
    }
    fits = {
        **refused,
        "content": {"nodes": [{"id": "a", "label": "Embodied agent", "kind": "process"},
                              {"id": "b", "label": "Action", "kind": "process"}],
                    "edges": [{"from": "a", "to": "b"}]},
    }
    prompts: list[str] = []

    def fake_call(prompt, **kwargs):
        calls["n"] += 1
        prompts.append(prompt)
        return json.dumps(refused if calls["n"] == 1 else fits)

    monkeypatch.setattr(module, "_call_model", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    receipt = module.generate(deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "out",
                              critic=False)
    row = receipt["specs"][0]
    assert row["compile_feedback"]["repaired"] is True
    assert "THE COMPILER REFUSED YOUR SPEC" in prompts[-1]
    assert calls["n"] == 2
    assert receipt["status"] == "pass"


def test_authoring_calls_disable_extended_thinking(monkeypatch) -> None:
    """Left at the provider's default these calls spent the whole budget inside
    a thinking block and returned no text: 8000 tokens, 90s, nothing. Raising
    the ceiling did not help; disabling it answered the same prompt in 11s."""

    from conference_pipeline.model_governance import default_model, llm_client

    module = _load("generate_figure_specs.py")
    seen: dict[str, object] = {}

    class _Block:
        type = "text"
        text = "{}"

    class _Result:
        stop_reason = "end_turn"
        content = [_Block()]

    def fake_create(client, **kwargs):
        seen.update(kwargs)
        return _Result()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "_anthropic_client", lambda request: object())
    monkeypatch.setattr(llm_client, "anthropic_messages_create", fake_create)
    module._call_model("prompt", model=default_model("presentation.figure_spec_authoring"), caller="c")
    assert seen["thinking"] == {"type": "disabled"}


def test_an_empty_response_is_reported_as_a_budget_problem(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    receipt = module.generate(
        deck_dir=_critic_deck(tmp_path), workspace=tmp_path, spec_dir=tmp_path / "out",
    )
    assert receipt["status"] == "blocked"
    assert any("empty response" in error
               for blocker in receipt["blockers"] for error in blocker["errors"])


def _numbered_deck(tmp_path: Path, count: int = 6) -> Path:
    """A deck of independent slides, so ordering is the only thing under test."""

    deck = tmp_path / "deck"
    deck.mkdir(exist_ok=True)
    (deck / "presentation_deck.json").write_text(json.dumps({
        "slides": [
            {"slide_id": f"sl_{i:02d}_slide", "section_id": "S1", "title": f"Slide {i}",
             "visual_role": "mechanism_pipeline: stages | components: A; B | edges: A -> B",
             "claims": [f"Claim {i}."], "visible_text": [f"Point {i}"]}
            for i in range(1, count + 1)
        ],
    }), encoding="utf-8")
    (deck / "speaker_author_work_ledger.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    return deck


def test_authoring_in_parallel_gives_the_same_receipt_as_one_at_a_time(tmp_path: Path, monkeypatch) -> None:
    """Threads must not decide what the deck says: same specs, same order, same
    archetypes, whatever order the calls come back in."""

    import random as _random

    module = _load("generate_figure_specs.py")
    deck = _numbered_deck(tmp_path)

    def fake_call(prompt, **kwargs):
        # Returning out of order is the point: sleep by a value that has
        # nothing to do with slide order.
        time.sleep(_random.uniform(0.0, 0.05))
        slide_id = [l for l in prompt.splitlines() if l.strip().startswith("slide_id:")][0].split()[-1]
        return json.dumps({
            "spec_version": module.SPEC_VERSION, "figure_id": f"{slide_id}-figure",
            "slide_id": slide_id, "archetype": "flow", "claim": "A leads to B.",
            "takeaway": "A leads to B", "grounding": {"kind": "schematic"},
            "content": {"nodes": [{"id": "a", "label": "A", "kind": "process"},
                                  {"id": "b", "label": "B", "kind": "process"}],
                        "edges": [{"from": "a", "to": "b"}]},
        })

    monkeypatch.setattr(module, "_call_model", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    serial = module.generate(deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "a", workers=1)
    parallel = module.generate(deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "b", workers=6)

    def shape(receipt):
        return [(row["slide_id"], row["archetype"], row["selection"]["archetype"])
                for row in receipt["specs"]]

    assert shape(serial) == shape(parallel)
    assert [s["slide_id"] for s in parallel["specs"]] == sorted(s["slide_id"] for s in parallel["specs"])
    assert parallel["status"] == serial["status"] == "pass"


def test_a_rate_limit_is_retried_rather_than_abandoning_the_deck(monkeypatch) -> None:
    """Several calls in flight makes a 429 ordinary. It used to propagate out
    of generate() and abandon every remaining slide."""

    module = _load("generate_figure_specs.py")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("rate_limit_error: too many requests")
        return "ok"

    assert module._with_backoff(flaky) == "ok"
    assert attempts["n"] == 3

    def broken():
        raise ValueError("malformed request")

    with pytest.raises(ValueError):
        module._with_backoff(broken)


def test_parallel_authoring_keeps_each_receipt_under_its_own_stage(tmp_path: Path, monkeypatch) -> None:
    """Receipts resolve their stage from a ContextVar first, and a worker thread
    inherits neither that nor a safe copy of the environment the critic mutates."""

    from conference_pipeline.provider_receipts import (
        current_provider_receipt_context,
        provider_receipt_context,
    )

    module = _load("generate_figure_specs.py")
    deck = _numbered_deck(tmp_path, count=6)
    seen: list[str] = []
    lock = threading.Lock()

    def fake_call(prompt, **kwargs):
        with lock:
            seen.append(str(current_provider_receipt_context().stage_id or ""))
        # A concurrent critic call sets the process-wide variables mid-flight.
        with provider_receipt_context(workspace_root=tmp_path, stage_id="figure_spec_critic",
                                      task_id="t", caller="c"):
            time.sleep(0.01)
        slide_id = [l for l in prompt.splitlines() if l.strip().startswith("slide_id:")][0].split()[-1]
        return json.dumps({
            "spec_version": module.SPEC_VERSION, "figure_id": f"{slide_id}-figure",
            "slide_id": slide_id, "archetype": "flow", "claim": "A leads to B.",
            "takeaway": "A leads to B", "grounding": {"kind": "schematic"},
            "content": {"nodes": [{"id": "a", "label": "A", "kind": "process"},
                                  {"id": "b", "label": "B", "kind": "process"}],
                        "edges": [{"from": "a", "to": "b"}]},
        })

    monkeypatch.setattr(module, "_call_model", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with provider_receipt_context(workspace_root=tmp_path, stage_id="figure_spec_authoring",
                                  task_id=module.TASK_ID, caller="c"):
        module.generate(deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "out", workers=6,
                        critic=False)

    assert seen and set(seen) == {"figure_spec_authoring"}, seen


def test_a_slide_is_not_offered_another_slides_numbers(deck: Path) -> None:
    """Drop-proof for the source binding: sl_01 does not cite SRC-014, so the
    ledger's rows are not its evidence however well the words match."""

    module = _load("generate_figure_specs.py")
    receipt = module.generate(deck_dir=deck, workspace=deck.parent.parent, dry_run=True)
    by_slide = {row["slide_id"]: row for row in receipt["specs"]}
    assert by_slide["sl_02_pruning"]["ledger_rows_offered"]
    assert by_slide["sl_01_intro"]["ledger_rows_offered"] == []
    prompt = (deck / "figure_specs" / "sl_01_intro-figure.prompt.txt").read_text(encoding="utf-8")
    assert "80.2" not in prompt
    assert "no numeric-claims ledger row matches this slide" in prompt


def test_a_spec_the_compiler_still_refuses_is_a_blocker_not_a_pass(tmp_path: Path, monkeypatch) -> None:
    """Deck 05's slide 25 was written, refused, repaired into the same refusal,
    and reported as 25/25 pass -- the deck compile found it two steps later."""

    module = _load("generate_figure_specs.py")
    refused = {
        "spec_version": module.SPEC_VERSION, "figure_id": "f", "slide_id": "s",
        "archetype": "flow", "claim": "Agents act on what they perceive.",
        "takeaway": "Perceive then act", "grounding": {"kind": "schematic"},
        "content": {"nodes": [{"id": "a", "label": "what the embodied agents actually are", "kind": "process"},
                              {"id": "b", "label": "Action", "kind": "process"}],
                    "edges": [{"from": "a", "to": "b"}]},
    }
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: json.dumps(refused))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    receipt = module.generate(
        deck_dir=_critic_deck(tmp_path), workspace=tmp_path, spec_dir=tmp_path / "out",
    )
    assert receipt["status"] == "blocked"
    assert [b["code"] for b in receipt["blockers"]] == ["figure_spec_refused_by_compiler"]


def test_the_repair_prompt_tells_a_mismatched_spec_to_change_archetype(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    loop_spec = {
        "spec_version": module.SPEC_VERSION, "figure_id": "f", "slide_id": "s",
        "archetype": "loop", "claim": "It returns to its start.", "takeaway": "It loops",
        "grounding": {"kind": "schematic"},
        "content": {"nodes": [{"id": "a", "label": "Draft"}, {"id": "b", "label": "Check"},
                              {"id": "c", "label": "Revise"}],
                    "unreadable_key": ["No weight updates"]},
    }
    prompts: list[str] = []

    def fake_call(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(loop_spec)

    monkeypatch.setattr(module, "_call_model", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    module.generate(deck_dir=_critic_deck(tmp_path), workspace=tmp_path, spec_dir=tmp_path / "out")
    repair = [p for p in prompts if "THE COMPILER REFUSED YOUR SPEC" in p]
    if repair:
        assert "Choose an archetype that READS" in repair[-1]


def test_author_and_critic_see_the_approved_sources_and_what_each_supports(tmp_path: Path) -> None:
    """Without the evidence the author invented a defense ('ToolShield') and the
    critic could not tell a real method name from an invented one."""

    module = _load("generate_figure_specs.py")
    deck = _critic_deck(tmp_path)
    (deck / "presentation_blueprint.json").write_text(json.dumps({"slides": [{
        "slide_id": "sl_01_fusion",
        "narration_beats": ["CausalArmor checks causal attribution before a tool call."],
        "source_plan": [{"label": "SRC-022", "supports": "CausalArmor blocks injected tool calls."}],
        "speaker_author_work_mentions": [{"label": "SRC-022", "source_id": "paper_abstract:c",
                                          "title": "CausalArmor", "authors": ["Ann Lee"]}],
    }]}), encoding="utf-8")
    out = tmp_path / "out"
    module.generate(deck_dir=deck, workspace=tmp_path, dry_run=True, spec_dir=out)
    prompt = next(out.glob("*.prompt.txt")).read_text(encoding="utf-8")
    assert '"supports": "CausalArmor blocks injected tool calls."' in prompt
    assert '"title": "CausalArmor"' in prompt and '"authors": ["Ann Lee"]' in prompt
    assert "approved narration: CausalArmor checks causal attribution" in prompt
    assert "Never introduce another name" in prompt

    slides = [{"slide_id": "sl_01_fusion"}]
    module.attach_approved_sources(slides, json.loads((deck / "presentation_blueprint.json").read_text()))
    evidence = module.critic_evidence(slides[0], [])
    for needle in ('SOURCE BLOCK', '"label": "SRC-022"', '"title": "CausalArmor"', '"authors": ["Ann Lee"]',
                   '"supports": "CausalArmor blocks injected tool calls."'):
        assert needle in evidence
    critic_prompt = module.build_critic_prompt(payload={}, slide=slides[0], archetype="flow", evidence=evidence)
    assert "contribution_note and work_kind, never against the slide claim" in critic_prompt


def test_a_spec_only_over_the_authoring_budget_is_kept_when_it_compiles(tmp_path: Path, monkeypatch) -> None:
    """The budget is a floor, not a boundary: five ICML region figures were
    refused for labels 1-6 characters over without ever being drawn."""

    module = _load("generate_figure_specs.py")
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: json.dumps(_good_flow_spec(module)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    over = "label 'Speed: throughput...' is 31 characters; at 4 items 'flow' allows 26."
    monkeypatch.setattr(module, "validate_spec", lambda spec, mode="authoring": [] if mode == "compile" else [over])
    deck = _critic_deck(tmp_path)
    receipt = module.generate(deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "out", critic=False)
    assert receipt["status"] == "pass"
    assert receipt["specs"][0]["budget_overrun_compiled"] == [over]

    def refuses(spec):
        raise module.FigureCompilerError("does not fit")

    monkeypatch.setattr(module, "compile_figure", refuses)
    receipt = module.generate(deck_dir=deck, workspace=tmp_path, spec_dir=tmp_path / "out2", critic=False)
    assert receipt["blockers"][0]["code"] == "figure_spec_invalid"


def test_a_revision_that_will_not_draw_gets_one_refusal_fed_back(tmp_path: Path, monkeypatch) -> None:
    """Deck 01/02: 29 of 79 critic rewrites were discarded by our own layout
    gates, finding and all; the revision now gets the refusal back once."""

    module = _load("generate_figure_specs.py")
    good = _good_flow_spec(module)
    calls: list[str] = []

    def fake_call(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(good)

    compiles = {"n": 0}
    real_compile = module.compile_figure

    def flaky_compile(spec):
        compiles["n"] += 1
        # the first compile is the revision's (initial authoring validates only);
        # it is refused, the fed-back retry draws, and so does the final compile
        if compiles["n"] == 1:
            raise module.FigureCompilerError("label 'A' overlaps 'B' by 4px")
        return real_compile(spec)

    monkeypatch.setattr(module, "_call_model", fake_call)
    monkeypatch.setattr(module, "compile_figure", flaky_compile)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    verdicts = [_major("overclaim"), {"verdict": "accept", "issues": []}]
    receipt = module.generate(
        deck_dir=_critic_deck(tmp_path), workspace=tmp_path, spec_dir=tmp_path / "out",
        critic_runner=lambda payload, slide, archetype: verdicts.pop(0),
    )
    row = receipt["specs"][0]["critic"]
    assert row["reauthored"] is True and row["unresolved_major"] == []
    assert "overlaps" in row["rounds"][0]["revision_feedback"]
    assert any("YOUR REVISION WAS REFUSED BEFORE IT COULD BE DRAWN" in p for p in calls)


def test_the_author_is_shown_only_the_brief_components_the_slide_supports(tmp_path: Path) -> None:
    """Deck 02 slide 8: 'Hand-designed harness' came from the brief and was
    drawn because the prompt said to draw the brief's components."""

    module = _load("generate_figure_specs.py")
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "presentation_deck.json").write_text(json.dumps({"slides": [{
        "slide_id": "sl_08_orchestration",
        "title": "Orchestration as generation",
        "claims": ["AOrchestra and OpenSage generate agent structure instead of hand-authoring it."],
        "visible_text": ["AOrchestra automates sub-agent creation", "OpenSage generates topology, toolset and memory"],
        "visual_role": "comparison_matrix: Hand-designed harness vs AOrchestra vs OpenSage | components: "
                       "Hand-designed harness; AOrchestra; OpenSage; Sub-agent creation; Topology | "
                       "edges: AOrchestra -> Sub-agent creation; Hand-designed harness -> Topology",
    }]}), encoding="utf-8")
    out = tmp_path / "out"
    receipt = module.generate(deck_dir=deck, workspace=tmp_path, dry_run=True, spec_dir=out)
    prompt = next(out.glob("*.prompt.txt")).read_text(encoding="utf-8")
    assert "    - AOrchestra\n" in prompt and "    - Sub-agent creation\n" in prompt
    assert "NOT supported by this slide's sources" in prompt and "    - Hand-designed harness" in prompt
    assert "Hand-designed harness -> Topology" not in prompt
    assert "Draw the supported components" in prompt and "MUST appear verbatim" in prompt
    assert receipt["specs"][0]["brief_components_dropped"] == ["Hand-designed harness"]


def test_source_plan_notes_survive_empty_binding_fields(tmp_path: Path) -> None:
    """Binding identity wins without erasing the slide's supports or source note."""
    module = _load("generate_figure_specs.py")
    deck = _critic_deck(tmp_path)
    plan = {"label": "SRC-022", "supports": "CausalArmor blocks injected tool calls.",
            "contribution_note": "Introduces causal attribution before agent tool execution.",
            "work_kind": "method"}
    mention = {"label": "SRC-022", "source_id": "paper_abstract:c", "title": "CausalArmor",
               "authors": ["Ann Lee"], "contribution_note": ""}
    blueprint_slide = {"slide_id": "sl_01_fusion", "source_plan": [plan],
                       "speaker_author_work_mentions": [mention]}
    # Direct blueprint consumers also need source_plan evidence, before attach.
    direct = module.approved_source_lines(blueprint_slide)
    assert plan["supports"] in direct and plan["contribution_note"] in direct
    (deck / "presentation_blueprint.json").write_text(json.dumps({"slides": [blueprint_slide]}))
    (deck / "source_binding_report.json").write_text(json.dumps({"slides": [{
        "slide_id": "sl_01_fusion", "visible_references": [{**mention, "supports": ""}],
    }]}))
    out = tmp_path / "out"
    module.generate(deck_dir=deck, workspace=tmp_path, dry_run=True, spec_dir=out)
    prompt = next(out.glob("*.prompt.txt")).read_text()
    for value in (plan["supports"], plan["contribution_note"], "CausalArmor", "Ann Lee", '"work_kind": "method"'):
        assert value in prompt


def test_bound_source_note_overrides_blueprint_note(tmp_path: Path) -> None:
    module = _load("generate_figure_specs.py")
    deck = _critic_deck(tmp_path)
    mention = {"label": "SRC-022", "source_id": "paper_abstract:c", "title": "CausalArmor",
               "authors": ["Ann Lee"], "contribution_note": "Outdated source note."}
    (deck / "presentation_blueprint.json").write_text(json.dumps({"slides": [{
        "slide_id": "sl_01_fusion", "speaker_author_work_mentions": [mention],
        "source_plan": [{"label": "SRC-022", "supports": "CausalArmor blocks injected tool calls."}],
    }]}))
    (deck / "source_binding_report.json").write_text(json.dumps({"slides": [{
        "slide_id": "sl_01_fusion", "visible_references": [{**mention, "contribution_note": "Current bound source note."}],
    }]}))
    out = tmp_path / "out"
    module.generate(deck_dir=deck, workspace=tmp_path, dry_run=True, spec_dir=out)
    prompt = next(out.glob("*.prompt.txt")).read_text()
    assert "Current bound source note." in prompt and "Outdated source note." not in prompt
    assert "CausalArmor blocks injected tool calls." in prompt


def _cached_duplicate_heading(tmp_path: Path, module):
    deck_dir = _critic_deck(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    old = {**_good_flow_spec(module), "title": "From images to video"}
    path = out / "sl_01_fusion-figure.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    return deck_dir, out, path, old


def _no_model(*args, **kwargs):
    raise AssertionError("an accepted or operator-patched spec is never re-authored")


def test_an_accepted_spec_is_not_reopened_by_a_later_lint(tmp_path: Path, monkeypatch) -> None:
    # ICML 2026 (2026-11-06): lints added after a deck was approved re-authored its
    # untouched specs on every resume, and each fresh call changed reviewed content.
    module = _load("generate_figure_specs.py")
    deck_dir, out, path, old = _cached_duplicate_heading(tmp_path, module)
    (deck_dir / module.ACCEPTED_DIGESTS_NAME).write_text(json.dumps({"sl_01_fusion-figure": module.spec_digest(old)}))
    monkeypatch.setattr(module, "_call_model", _no_model)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    receipt = module.generate(deck_dir=deck_dir, workspace=tmp_path, spec_dir=out, skip_existing=True, critic=False)
    assert receipt["status"] == "pass"
    assert json.loads(path.read_text()) == old
    assert receipt["specs"][0]["lint_findings_on_accepted_spec"]


def test_an_operator_accept_counts_as_accepted(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    deck_dir, out, path, old = _cached_duplicate_heading(tmp_path, module)
    (out / "sl_01_fusion-figure.patch.json").write_text(json.dumps([
        {"outcome": "accept", "before_sha256": module.spec_digest(old), "by": "Operator", "reason": "stands"}]))
    monkeypatch.setattr(module, "_call_model", _no_model)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert module.generate(deck_dir=deck_dir, workspace=tmp_path, spec_dir=out, skip_existing=True, critic=False)["status"] == "pass"


def test_an_operator_patch_that_fails_a_lint_blocks_instead_of_being_overwritten(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    deck_dir, out, path, old = _cached_duplicate_heading(tmp_path, module)
    (out / "sl_01_fusion-figure.patch.json").write_text(json.dumps([
        {"outcome": "patch", "before_sha256": "x", "after_sha256": module.spec_digest(old), "by": "Operator", "reason": "edit"}]))
    monkeypatch.setattr(module, "_call_model", _no_model)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    receipt = module.generate(deck_dir=deck_dir, workspace=tmp_path, spec_dir=out, skip_existing=True, critic=False)
    assert receipt["status"] != "pass"
    assert receipt["blockers"][0]["code"] == "operator_patch_fails_lint" and receipt["blockers"][0]["errors"]
    assert json.loads(path.read_text()) == old, "the operator's edit is left for the operator"


def test_a_passing_spec_is_recorded_as_accepted(tmp_path: Path, monkeypatch) -> None:
    module = _load("generate_figure_specs.py")
    deck_dir, out, path, old = _cached_duplicate_heading(tmp_path, module)
    repaired = {**old, "title": "Temporal fusion mechanism"}
    monkeypatch.setattr(module, "_call_model", lambda *a, **k: json.dumps(repaired))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    receipt = module.generate(deck_dir=deck_dir, workspace=tmp_path, spec_dir=out, skip_existing=True, critic=False)
    accepted = json.loads((deck_dir / module.ACCEPTED_DIGESTS_NAME).read_text())
    assert accepted["sl_01_fusion-figure"] == receipt["specs"][0]["spec_sha256"] == module.spec_digest(json.loads(path.read_text()))
