from __future__ import annotations

import json
import os
from pathlib import Path

from conference_pipeline.presentation import step_lineage as sl
from conference_pipeline.presentation.step_lineage import DeckStepLineage, StepInput, StepSpec


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _deck(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    deck = workspace / "decks" / "01_topic"
    deck.mkdir(parents=True)
    return workspace, deck


def _spec(step_id: str, inputs: list[Path], outputs: list[Path], **kw) -> StepSpec:
    return StepSpec(
        step_id=step_id,
        inputs=tuple(StepInput(p) for p in inputs),
        outputs=tuple(outputs),
        **kw,
    )


def _run(ledger: DeckStepLineage, spec: StepSpec, produce) -> None:
    snapshot = ledger.snapshot_inputs(spec)
    produce()
    ledger.record(spec, snapshot)


def test_should_skip_requires_sidecar_then_matches(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "blueprint.json", '{"a": 1}')
    out = deck / "contract.json"
    spec = _spec("contract", [src], [out])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    assert ledger.should_skip(spec, resume=False).reason == "resume_disabled"
    assert ledger.should_skip(spec, resume=True).reason == "no_sidecar"
    _run(ledger, spec, lambda: _write(out, '{"status": "ok"}'))
    decision = ledger.should_skip(spec, resume=True)
    assert decision.skip and decision.reason == "current"
    ledger.note_skip(spec, decision)
    reloaded = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    assert reloaded.summary()["steps_reused"] == ["contract"]
    assert reloaded.should_skip(spec, resume=True).skip


def test_input_change_reruns_and_reports_path(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "blueprint.json", '{"a": 1}')
    out = deck / "contract.json"
    spec = _spec("contract", [src], [out])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, spec, lambda: _write(out, "{}"))
    _write(src, '{"a": 2}')
    decision = ledger.should_skip(spec, resume=True)
    assert not decision.skip and decision.reason == "input_changed:ws:decks/01_topic/blueprint.json"


def test_missing_required_output_and_missing_optional_input_appearing_later(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "blueprint.json", "{}")
    judge = deck / "blueprint_llm_judge.md"  # optional, absent at first
    out = deck / "review.json"
    spec = StepSpec("review", (StepInput(src), StepInput(judge, optional=True)), (out,))
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, spec, lambda: _write(out, '{"status": "pass"}'))
    assert ledger.should_skip(spec, resume=True).skip
    _write(judge, "judge says")
    assert ledger.should_skip(spec, resume=True).reason == "input_changed:ws:decks/01_topic/blueprint_llm_judge.md"
    judge.unlink()
    out.write_text("", encoding="utf-8")
    assert ledger.should_skip(spec, resume=True).reason == "outputs_missing:ws:decks/01_topic/review.json"


def test_strict_output_hand_edit_reruns_but_editable_output_is_adopted(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "render_spec.json", "{}")
    raw = deck / "engaging_narration_raw.json"
    md = deck / "script.ENGAGING.md"
    producer = StepSpec("narration", (StepInput(src),), (raw, md), editable_outputs=frozenset({raw, md}))
    consumer = _spec("rebuild_delivery", [raw], [deck / "delivery.json"])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, producer, lambda: (_write(raw, '{"v": 1}'), _write(md, "one")))
    _run(ledger, consumer, lambda: _write(deck / "delivery.json", "{}"))
    assert ledger.should_skip(producer, resume=True).skip
    assert ledger.should_skip(consumer, resume=True).skip

    _write(raw, '{"v": 2}')  # operator patched the narration source of truth
    decision = ledger.should_skip(producer, resume=True)
    assert decision.skip and decision.details["adopted"] == ["ws:decks/01_topic/engaging_narration_raw.json"]
    reloaded = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    history = reloaded._history("ws:decks/01_topic/engaging_narration_raw.json")
    assert history[-1]["adopted_operator_edit"] is True and history[-1]["step"] == "narration"
    assert reloaded.should_skip(consumer, resume=True).reason.startswith("input_changed:")

    # A strict (non-editable) output that was hand-edited re-runs its producer.
    strict = _spec("contract", [src], [deck / "contract.json"])
    _run(ledger, strict, lambda: _write(deck / "contract.json", "{}"))
    _write(deck / "contract.json", '{"edited": true}')
    assert ledger.should_skip(strict, resume=True).reason == "output_changed:ws:decks/01_topic/contract.json"


def test_operator_edited_output_survives_upstream_input_regeneration(tmp_path: Path) -> None:
    """Regression for the 2026-09-25 incident: bp_repair.py hand-repairs a
    blueprint (an editable_outputs surface) as a versioned patch. Later, an
    unrelated fix upstream (e.g. a corpus repair-findings correction) makes
    the blueprint's OWN input (research_cluster_plan.json) legitimately
    regenerate. should_skip() protected the blueprint's own hash from drift
    but not its recorded input's -- so the input's superseded history forced
    generate_editorial_blueprint to rerun on the next --resume, silently
    overwriting the hand repair with a fresh, unreviewed LLM generation. The
    resumed run destroyed presentation_blueprint.json (and cascaded into
    presentation_deck.json and figure_specs) for four decks before it was
    caught. An operator-edited output must stay pinned regardless of what its
    inputs do afterwards; only an explicit further hand edit, or deleting the
    output outright, may move it again."""
    workspace, deck = _deck(tmp_path)
    upstream_out = deck / "research_cluster_plan.json"
    upstream = _spec("build_research_cluster_plan", [deck / "source_repair_findings.json"], [upstream_out])
    blueprint = deck / "presentation_blueprint.json"
    producer = StepSpec(
        "generate_editorial_blueprint",
        (StepInput(upstream_out),),
        (blueprint,),
        editable_outputs=frozenset({blueprint}),
    )
    # A plain (non-editable) consumer of the same upstream input: it proves
    # the upstream regeneration below actually superseded the recorded input
    # history, rather than the pin assertion passing for an unrelated reason.
    consumer = _spec("review_presentation_blueprint", [upstream_out], [deck / "blueprint_review.json"])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)

    # Deliberately different lengths (not just different bytes), so the
    # digest cache's size+mtime fast path can never mask a missed re-hash.
    findings = _write(deck / "source_repair_findings.json", '{"v": 1}')
    _run(ledger, upstream, lambda: _write(upstream_out, '{"plan": "first"}'))
    _run(ledger, producer, lambda: _write(blueprint, '{"revision": 1}'))
    _run(ledger, consumer, lambda: _write(deck / "blueprint_review.json", "{}"))
    assert ledger.should_skip(producer, resume=True).skip

    # Operator hand-repairs the blueprint (bp_repair.py's role): a versioned
    # patch, not a fresh generation.
    _write(blueprint, '{"revision": 2, "repair_of": "..."}')
    decision = ledger.should_skip(producer, resume=True)
    assert decision.skip and decision.details["adopted"] == ["ws:decks/01_topic/presentation_blueprint.json"]

    # An unrelated fix changes source_repair_findings.json, so its own
    # producer legitimately re-runs and rewrites research_cluster_plan.json --
    # superseding the blueprint's recorded input history.
    _write(findings, '{"v": 2, "extra_field_to_change_length": true}')
    _run(ledger, upstream, lambda: _write(upstream_out, '{"plan": "second, and longer"}'))

    # Control: an ordinary consumer of the same input is correctly forced to
    # re-run -- the incident precondition (a superseded input history) is
    # genuinely present at this point, not merely assumed.
    control = ledger.should_skip(consumer, resume=True)
    assert not control.skip
    assert control.reason == "input_changed:ws:decks/01_topic/research_cluster_plan.json"
    assert control.details == {"reason": "write_superseded"}

    # The repaired blueprint must stay pinned: it must not be forced to
    # regenerate and silently overwrite the hand repair.
    decision = ledger.should_skip(producer, resume=True)
    assert decision.skip and decision.reason == "pinned_by_operator_edit"

    # The pin must survive a fresh process loading the ledger from disk --
    # that is the actual protection a real ``--resume`` in a new invocation
    # relies on, not in-memory state left over from this test's own calls.
    reloaded = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    assert reloaded.should_skip(producer, resume=True).reason == "pinned_by_operator_edit"


def test_repair_marker_pins_output_after_ledger_rebuilt_from_disk(tmp_path: Path) -> None:
    """``--adopt-existing-outputs`` rebuilds a ledger from disk without the
    adopted-edit flag, so the pin must also come from the artifact: a
    blueprint carrying ``repair_of`` stays pinned when an input moves, while a
    fresh generation (no marker) still re-runs."""
    workspace, deck = _deck(tmp_path)
    plan = _write(deck / "research_cluster_plan.json", '{"plan": "first"}')
    blueprint = deck / "presentation_blueprint.json"
    producer = StepSpec(
        "generate_editorial_blueprint",
        (StepInput(plan),),
        (blueprint,),
        editable_outputs=frozenset({blueprint}),
    )

    _write(blueprint, '{"revision": 3, "repair_of": "abc123"}')
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    assert ledger.adopt(producer)
    history = ledger._history("ws:decks/01_topic/presentation_blueprint.json")
    assert not history[-1].get("adopted_operator_edit")  # the bootstrap drops the ledger flag
    _write(plan, '{"plan": "second, and longer"}')
    decision = DeckStepLineage.for_deck_dir(deck, workspace=workspace).should_skip(producer, resume=True)
    assert decision.skip and decision.reason == "pinned_by_operator_edit"

    fresh_deck = tmp_path / "fresh"
    fresh_ws, fresh = _deck(fresh_deck)
    fresh_plan = _write(fresh / "research_cluster_plan.json", '{"plan": "first"}')
    fresh_blueprint = _write(fresh / "presentation_blueprint.json", '{"revision": 1, "repair_of": null}')
    fresh_producer = StepSpec(
        "generate_editorial_blueprint",
        (StepInput(fresh_plan),),
        (fresh_blueprint,),
        editable_outputs=frozenset({fresh_blueprint}),
    )
    fresh_ledger = DeckStepLineage.for_deck_dir(fresh, workspace=fresh_ws)
    assert fresh_ledger.adopt(fresh_producer)
    _write(fresh_plan, '{"plan": "second, and longer"}')
    assert fresh_ledger.should_skip(fresh_producer, resume=True).reason == (
        "input_changed:ws:decks/01_topic/research_cluster_plan.json"
    )


def test_operator_edited_output_stays_pinned_after_downstream_in_place_edit(tmp_path: Path) -> None:
    """The narration-lane shape: after an editable output is adopted, a
    recorded ``in_place`` editor legitimately appends another write to that
    same path (e.g. narration's ``#post-gates --sync-raw-from-md`` step,
    which is itself in_place and reads/writes the same raw JSON). That write
    displaces the adopted entry from the *head* of the path's history. The
    pin must still hold on the next resume even though the adopted entry is
    no longer the head -- only a non-in-place reproduction, or a fresh hand
    edit, may reopen the step."""
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "render_spec.json", "{}")
    raw = deck / "engaging_narration_raw.json"
    producer = StepSpec("generate_engaging_narration", (StepInput(src),), (raw,), editable_outputs=frozenset({raw}))
    # An in_place editor that reads and rewrites the same path (mirrors
    # generate_engaging_narration's own #post-gates sync step).
    syncer = StepSpec("generate_engaging_narration#post-gates", (StepInput(raw),), (raw,), in_place=True)
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)

    _run(ledger, producer, lambda: _write(raw, '{"v": 1}'))
    assert ledger.should_skip(producer, resume=True).skip

    # Operator hand-patches the narration source of truth.
    _write(raw, '{"v": "operator patched"}')
    decision = ledger.should_skip(producer, resume=True)
    assert decision.skip and decision.reason == "pinned_by_operator_edit"

    # The in-place syncer runs (its job: keep sibling markdowns/JSON forms in
    # sync), appending a write onto the same path and displacing the adopted
    # entry from the head.
    _run(ledger, syncer, lambda: _write(raw, '{"v": "operator patched", "synced": true}'))
    history = ledger._history("ws:decks/01_topic/engaging_narration_raw.json")
    assert history[-1].get("adopted_operator_edit") is not True  # sanity: head moved

    # An unrelated upstream fix (e.g. a figure/render_spec correction) would,
    # pre-fix, have forced generate_engaging_narration to rerun here and
    # overwritten both the operator's patch and the syncer's work.
    _write(src, '{"changed": true}')
    decision = ledger.should_skip(producer, resume=True)
    assert decision.skip and decision.reason == "pinned_by_operator_edit"


def test_in_place_editor_is_transparent_upstream_and_opaque_downstream(tmp_path: Path) -> None:
    """The bookends worked example: a reader before the editor keeps skipping; a
    reader after the editor re-runs only when the editor's output changed."""

    workspace, deck = _deck(tmp_path)
    contract = _write(deck / "presentation_deck.json", "{}")
    spec_json = deck / "render_spec.json"
    html = deck / "presentation.html"
    render_spec = _spec("generate_render_spec", [contract], [spec_json])
    bookends = _spec("add_bookend_slides", [spec_json], [spec_json], in_place=True)
    render = _spec("md_to_revealjs", [spec_json], [html])
    binder = _spec("bind_sources", [contract], [deck / "binding.json"])  # reads only the contract, before bookends
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)

    _run(ledger, binder, lambda: _write(deck / "binding.json", "{}"))
    _run(ledger, render_spec, lambda: _write(spec_json, '{"slides": 1}'))
    _run(ledger, bookends, lambda: _write(spec_json, '{"slides": 1, "bookends": true}'))
    _run(ledger, render, lambda: _write(html, "<html/>"))

    history = ledger._history("ws:decks/01_topic/render_spec.json")
    assert [w["step"] for w in history] == ["generate_render_spec", "add_bookend_slides"]
    assert history[1]["in_place"] is True
    # Everyone skips on a clean resume, including the reader that ran BEFORE the in-place edit.
    for spec in (binder, render_spec, bookends, render):
        assert ledger.should_skip(spec, resume=True).skip, spec.step_id

    # Bookends re-runs and yields the SAME bytes: nothing downstream moves.
    snapshot = ledger.snapshot_inputs(bookends)
    ledger.record(bookends, snapshot)
    assert ledger.should_skip(render, resume=True).skip

    # Bookends re-runs and yields DIFFERENT bytes: render re-runs, render_spec does not.
    _run(ledger, bookends, lambda: _write(spec_json, '{"slides": 2, "bookends": true}'))
    assert ledger.should_skip(render_spec, resume=True).skip
    assert ledger.should_skip(render, resume=True).reason == "input_changed:ws:decks/01_topic/render_spec.json"
    # An in-place editor excludes its self-edited path from its own input check.
    assert ledger.should_skip(bookends, resume=True).skip

    # A second in-place editor after render (reconcile re-stamping hashes) is
    # transparent to render, and its own re-run stays transparent too.
    _run(ledger, render, lambda: _write(html, "<html>2</html>"))
    reconcile = _spec("reconcile_contract_hashes", [spec_json], [spec_json], in_place=True)
    _run(ledger, reconcile, lambda: _write(spec_json, '{"slides": 2, "bookends": true, "hashes": 1}'))
    assert ledger.should_skip(render, resume=True).skip
    _run(ledger, reconcile, lambda: _write(spec_json, '{"slides": 2, "bookends": true, "hashes": 2}'))
    assert ledger.should_skip(render, resume=True).skip
    assert [w["step"] for w in ledger._history("ws:decks/01_topic/render_spec.json")] == [
        "generate_render_spec",
        "add_bookend_slides",
        "reconcile_contract_hashes",
    ]
    # But bookends re-running with new bytes supersedes reconcile's write too.
    _run(ledger, bookends, lambda: _write(spec_json, '{"slides": 9, "bookends": true}'))
    assert [w["step"] for w in ledger._history("ws:decks/01_topic/render_spec.json")] == [
        "generate_render_spec",
        "add_bookend_slides",
    ]
    assert not ledger.should_skip(render, resume=True).skip

    # The producer re-runs: history resets, and the editor + render re-run.
    _run(ledger, render_spec, lambda: _write(spec_json, '{"slides": 3}'))
    assert [w["step"] for w in ledger._history("ws:decks/01_topic/render_spec.json")] == ["generate_render_spec"]
    assert not ledger.should_skip(bookends, resume=True).skip
    assert not ledger.should_skip(render, resume=True).skip


def test_directory_input_pattern_ignores_sibling_writes_and_values_change_reruns(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    generated = deck / "assets" / "generated"
    _write(generated / "fig1.svg", "<svg/>")
    out = deck / "render_spec.json"
    spec = StepSpec("render_spec", (StepInput(generated, pattern="*.svg"),), (out,), values={"--deck-kind": "topic"})
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, spec, lambda: _write(out, "{}"))
    _write(generated / "qc" / "shot.png", "png")  # a later step writes a sibling
    assert ledger.should_skip(spec, resume=True).skip
    _write(generated / "fig2.svg", "<svg/>")
    assert ledger.should_skip(spec, resume=True).reason == "input_changed:ws:decks/01_topic/assets/generated"

    other = StepSpec("render_spec", spec.inputs, spec.outputs, values={"--deck-kind": "executive"})
    _run(ledger, spec, lambda: None)
    assert ledger.should_skip(other, resume=True).reason == "values_changed"


def test_reject_and_accept_statuses(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "in.json", "{}")
    out = deck / "review.json"
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    verdict = _spec("review", [src], [out], accept_statuses=frozenset({"pass"}))
    _run(ledger, verdict, lambda: _write(out, '{"status": "fail"}'))
    assert ledger.should_skip(verdict, resume=True).reason == "status_not_accepted:ws:decks/01_topic/review.json"
    _run(ledger, verdict, lambda: _write(out, '{"status": "pass"}'))
    assert ledger.should_skip(verdict, resume=True).skip

    lint = _spec("lint", [src], [out], reject_statuses=frozenset())
    _run(ledger, lint, lambda: _write(out, '{"status": "failed"}'))
    assert ledger.should_skip(lint, resume=True).skip  # a content verdict, not an execution failure
    default = _spec("default", [src], [out])
    _run(ledger, default, lambda: None)
    assert ledger.should_skip(default, resume=True).reason == "status_rejected:ws:decks/01_topic/review.json"


def test_large_file_stat_cache_only_above_threshold(tmp_path: Path, monkeypatch) -> None:
    workspace, deck = _deck(tmp_path)
    big = deck / "video.mp4"
    big.write_bytes(b"x" * 32)
    small = _write(deck / "contract.json", '{"n": 1}')
    monkeypatch.setattr(sl, "LARGE_FILE_STAT_CACHE_BYTES", 16)
    spec = _spec("video", [small], [big])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, spec, lambda: None)
    stat = big.stat()
    # Same size + mtime, different bytes: the large-file rule trusts the recorded digest ...
    sl._SESSION_DIGEST_CACHE.clear()
    big.write_bytes(b"y" * 32)
    os.utime(big, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert ledger.should_skip(spec, resume=True).skip
    # ... but a small contract is always re-hashed.
    sl._SESSION_DIGEST_CACHE.clear()
    small_stat = small.stat()
    small.write_text('{"n": 2}', encoding="utf-8")
    os.utime(small, ns=(small_stat.st_atime_ns, small_stat.st_mtime_ns))
    assert ledger.should_skip(spec, resume=True).reason == "input_changed:ws:decks/01_topic/contract.json"


def test_output_digest_ledger_relocatable_and_stage_hook(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "in.json", "{}")
    out = deck / "out.json"
    spec = _spec("s", [src], [out])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    before = ledger.output_digest()
    _run(ledger, spec, lambda: _write(out, "{}"))
    after = ledger.output_digest()
    assert before != after
    assert ledger.ledger_sha256() and not ledger.ledger_path.with_name(".step_lineage.json.tmp").exists()
    data = json.loads(ledger.ledger_path.read_text(encoding="utf-8"))
    assert all(key.startswith("ws:") for key in data["paths"])

    moved_root = tmp_path / "elsewhere" / "demo2026"
    moved_root.parent.mkdir(parents=True)
    os.rename(workspace, moved_root)
    relocated = DeckStepLineage.for_deck_dir(moved_root / "decks" / "01_topic", workspace=moved_root)
    assert relocated.output_digest() == after
    assert relocated.should_skip(_spec("s", [moved_root / "decks" / "01_topic" / "in.json"], [moved_root / "decks" / "01_topic" / "out.json"]), resume=True).skip
    assert sl.presentation_output_digest({"01_topic": relocated}) == sl.canonical_json_digest({"01_topic": after})
    assert sl.load_deck_output_digests(moved_root, [moved_root / "decks" / "01_topic"]) == {"01_topic": after}


def test_adopt_bootstraps_only_complete_steps(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "in.json", "{}")
    done = _spec("done", [src], [_write(deck / "done.json", "{}")])
    missing = _spec("missing", [src], [deck / "never.json"])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    assert ledger.adopt(done) is True and ledger.adopt(done) is False
    assert ledger.adopt(missing) is False
    assert ledger.summary()["steps_adopted"] == ["done"]
    assert ledger.should_skip(done, resume=True).skip


def test_from_command_derives_inputs_values_and_excludes_outputs(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    blueprint = _write(deck / "presentation_blueprint.json", "{}")
    ledger_json = _write(deck / "speaker_author_work_ledger.json", "{}")
    brand = _write(tmp_path / "repo" / "assets" / "brand" / "logo.png", "png")
    out = deck / "presentation_deck.json"
    command = [
        "python",
        "scripts/presentations/generate_deck_contract.py",
        "--workspace", str(workspace),
        "--blueprint", str(blueprint),
        f"--ledger={ledger_json}",
        "--output", str(out),
        "--deck-kind", "topic",
        "--strict",
        "--missing-input", str(deck / "absent.json"),
    ]
    spec = StepSpec.from_command(
        command,
        workspace=workspace,
        outputs=[out],
        output_flags=["--output"],
        input_flags=["--missing-input"],
        extra_inputs=[StepInput(brand)],
        repo_root=tmp_path / "repo",
        values={"deck": "01"},
        accept_statuses=frozenset({"pass"}),
    )
    assert spec.step_id == "generate_deck_contract"
    assert [p.path for p in spec.inputs] == [blueprint.resolve(), ledger_json.resolve(), deck / "absent.json", brand.resolve()]
    assert spec.inputs[2].optional is True and spec.inputs[0].optional is False
    assert spec.outputs == (out.resolve(),)
    assert spec.values == {"--deck-kind": "topic", "--strict": True, "deck": "01"}
    assert spec.accept_statuses == frozenset({"pass"})
    assert spec.command[1].endswith("generate_deck_contract.py")


def test_record_presentation_command_round_trip(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    src = _write(deck / "render_spec.json", "{}")
    out = deck / "presentation.html"
    command = ["python", "md_to_revealjs.py", "--render-spec", str(src), "--output", str(out)]
    _write(out, "<html/>")
    spec = sl.record_presentation_command(command, workspace=workspace, deck_dir=deck, outputs=[out], output_flags=["--output"])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    assert ledger.should_skip(spec, resume=True).skip
    _write(src, '{"changed": 1}')
    assert not ledger.should_skip(spec, resume=True).skip


def test_in_place_editor_reruns_when_its_edited_output_was_reproduced(tmp_path: Path) -> None:
    """Figure pass 2 edits pass 1's plan in place; pass 1 re-running must re-run pass 2.

    The editor declares the plan only as an output (argv-derived inputs never
    include a step's own outputs), so the check has to live on the outputs.
    """

    workspace, deck = _deck(tmp_path)
    brief = _write(deck / "brief.json", '{"topic": 1}')
    plan = deck / "visual_plan.json"
    pass1 = _spec("figures_pass1", [brief], [plan])
    pass2 = _spec("figures_pass2", [], [plan], in_place=True)
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, pass1, lambda: _write(plan, '{"figures": "placeholder"}'))
    _run(ledger, pass2, lambda: _write(plan, '{"figures": "generated"}'))
    assert ledger.should_skip(pass1, resume=True).skip
    assert ledger.should_skip(pass2, resume=True).skip

    # pass 1 re-runs (its brief changed) and puts placeholders back
    _write(brief, '{"topic": 2}')
    assert not ledger.should_skip(pass1, resume=True).skip
    _run(ledger, pass1, lambda: _write(plan, '{"figures": "placeholder v2"}'))
    decision = ledger.should_skip(pass2, resume=True)
    assert not decision.skip
    assert decision.reason == "output_changed:ws:decks/01_topic/visual_plan.json"
    assert decision.details == {"reason": "write_superseded"}

    # re-running pass 2 makes it current again; a later in-place edit by
    # another editor keeps it current
    _run(ledger, pass2, lambda: _write(plan, '{"figures": "generated v2"}'))
    assert ledger.should_skip(pass2, resume=True).skip
    polish = _spec("polish", [], [plan], in_place=True)
    _run(ledger, polish, lambda: _write(plan, '{"figures": "generated v2", "polished": true}'))
    assert ledger.should_skip(pass2, resume=True).skip


def test_in_place_editor_byte_identical_write_stays_current(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    plan = deck / "visual_plan.json"
    pass1 = _spec("figures_pass1", [], [plan])
    pass2 = _spec("figures_pass2", [], [plan], in_place=True)
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, pass1, lambda: _write(plan, '{"figures": "final"}'))
    _run(ledger, pass2, lambda: _write(plan, '{"figures": "final"}'))  # no-op edit
    assert ledger.should_skip(pass2, resume=True).skip


# Per-command (argv-driven) resume, as the ICLR/ICML runners port it: the step
# is keyed from the command line, recorded with ``record_presentation_command``
# and skipped only while the deck ledger digests still match. These replace the
# ICML runner tests of its retired local ``presentation_command_*`` wrappers.
_COMMAND_OUTPUT_FLAGS = frozenset({"--output-slides-md"})
_COMMAND_INPUT_FLAGS = frozenset({"--analysis-md"})


def _command_skip(command: list[str], *, workspace: Path, deck: Path, output: Path) -> bool:
    spec = StepSpec.from_command(
        command,
        workspace=workspace,
        outputs=[output],
        output_flags=_COMMAND_OUTPUT_FLAGS,
        input_flags=_COMMAND_INPUT_FLAGS,
    )
    return DeckStepLineage.for_deck_dir(deck, workspace=workspace).should_skip(spec, resume=True).skip


def _record_command(command: list[str], *, workspace: Path, deck: Path, output: Path) -> None:
    sl.record_presentation_command(
        command,
        workspace=workspace,
        deck_dir=deck,
        outputs=[output],
        output_flags=_COMMAND_OUTPUT_FLAGS,
        input_flags=_COMMAND_INPUT_FLAGS,
    )


def test_command_resume_skip_requires_ledger_record(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    output = deck / "1_draft_slides.md"
    command = [
        "scripts/presentations/generate_slides_agent.py",
        "--workspace",
        str(workspace),
        "--output-slides-md",
        str(output),
    ]
    assert _command_skip(command, workspace=workspace, deck=deck, output=output) is False
    _write(output, "## Slide 1: Demo\n")
    assert _command_skip(command, workspace=workspace, deck=deck, output=output) is False
    _record_command(command, workspace=workspace, deck=deck, output=output)
    assert (deck / "step_lineage.json").is_file()
    assert _command_skip(command, workspace=workspace, deck=deck, output=output) is True


def test_command_resume_skip_rejects_changed_input_digest_not_mtime(tmp_path: Path) -> None:
    workspace, deck = _deck(tmp_path)
    input_path = _write(workspace / "analysis" / "slide_briefs" / "01_topic.md", "# brief\n")
    output = _write(deck / "1_draft_slides.md", "## Slide 1: Demo\n")
    command = [
        "scripts/presentations/generate_slides_agent.py",
        "--workspace",
        str(workspace),
        "--analysis-md",
        str(input_path),
        "--output-slides-md",
        str(output),
    ]
    _record_command(command, workspace=workspace, deck=deck, output=output)
    assert _command_skip(command, workspace=workspace, deck=deck, output=output) is True

    stat = input_path.stat()
    os.utime(input_path, ns=(stat.st_atime_ns + 10_000_000_000, stat.st_mtime_ns + 10_000_000_000))
    assert _command_skip(command, workspace=workspace, deck=deck, output=output) is True

    input_path.write_text("# brief, revised\n", encoding="utf-8")
    assert _command_skip(command, workspace=workspace, deck=deck, output=output) is False


def test_operator_named_step_reruns_even_when_current(tmp_path: Path) -> None:
    """--rerun-steps: a generator fix changes no recorded input, so only an
    explicit request can regenerate a current step; other steps still skip."""
    from conference_pipeline.production import presentation_steps as steps

    workspace, deck = _deck(tmp_path)
    src = _write(deck / "plan.json", '{"a": 1}')
    blueprint = deck / "blueprint.json"
    contract = deck / "contract.json"
    spec_blueprint = _spec("generate_editorial_blueprint", [src], [blueprint])
    spec_contract = _spec("contract", [src], [contract])
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, spec_blueprint, lambda: _write(blueprint, "{}"))
    _run(ledger, spec_contract, lambda: _write(contract, "{}"))

    lineage_token = steps.LINEAGE.set(ledger)
    try:
        assert steps.step_gate(spec_blueprint, resume=True)[0] is True
        rerun_token = steps.RERUN_STEPS.set(frozenset({"generate_editorial_blueprint"}))
        try:
            skip, snapshot = steps.step_gate(spec_blueprint, resume=True)
            assert skip is False and snapshot is not None
            assert steps.step_gate(spec_contract, resume=True)[0] is True
        finally:
            steps.RERUN_STEPS.reset(rerun_token)
        assert steps.step_gate(spec_blueprint, resume=True)[0] is True
    finally:
        steps.LINEAGE.reset(lineage_token)


def test_a_drifted_delivery_contract_is_archived_and_stubbed(tmp_path: Path) -> None:
    """A regenerated blueprint renames slides; the old delivery contract then
    blocked render-spec generation ('segment slide IDs drift from deck')."""
    import json as _json

    from conference_pipeline.production.presentation_steps import bootstrap_delivery_stub

    contract = {"presentation_id": "p", "slides": [{"slide_id": "sl_01_new"}, {"slide_id": "sl_02_new"}]}
    old = {"segments": [{"slide_id": "sl_01_old"}]}
    (tmp_path / "delivery_single_speaker.json").write_text(_json.dumps(old), encoding="utf-8")
    current = {"segments": [{"slide_id": "sl_01_new"}, {"slide_id": "sl_02_new"}], "keep": True}
    (tmp_path / "delivery_notebooklm.json").write_text(_json.dumps(current), encoding="utf-8")

    bootstrap_delivery_stub(deck_dir=tmp_path, deck_contract=contract)

    single = _json.loads((tmp_path / "delivery_single_speaker.json").read_text(encoding="utf-8"))
    assert [s["slide_id"] for s in single["segments"]] == ["sl_01_new", "sl_02_new"]
    archived = list(tmp_path.glob("delivery_single_speaker.stale-*.json"))
    assert len(archived) == 1 and _json.loads(archived[0].read_text(encoding="utf-8")) == old
    assert _json.loads((tmp_path / "delivery_notebooklm.json").read_text(encoding="utf-8")) == current


def test_a_long_comma_joined_value_is_not_mistaken_for_a_path(tmp_path: Path) -> None:
    """--figure-slide-ids joins slide ids into one argument; stat() on it raised
    'File name too long' and the whole stage crashed before running (2026-10-01)."""

    workspace, deck = _deck(tmp_path)
    ids = ",".join(f"sl_{i:02d}_{'x' * 40}" for i in range(12))
    spec = StepSpec.from_command(
        ["python", "scripts/x.py", "--slide-ids", ids, "--receipt", str(deck / "r.json")],
        workspace=workspace, outputs=[deck / "r.json"],
    )
    assert all(ids not in str(inp.path) for inp in spec.inputs)


def test_editing_a_file_inside_a_declared_input_directory_reruns_the_step(tmp_path: Path) -> None:
    """The shape the figure compile step now has: its inputs include a directory of
    hand-editable specs (pattern ``*-figure.json``) beside its own aggregate output."""

    workspace, deck = _deck(tmp_path)
    spec_dir = deck / "figure_specs"
    spec = _write(spec_dir / "sl_07-figure.json", '{"label": "Untested prediction"}')
    _write(spec_dir / "sl_07-figure.critic.json", '{"note": "sidecar"}')
    # Written by the authoring step before compile runs, and rewritten by compile.
    aggregate = _write(deck / "figure_specs.json", "{}")
    compile_step = StepSpec(
        step_id="compile_figures",
        inputs=(StepInput(spec_dir, pattern="*-figure.json"), StepInput(aggregate)),
        outputs=(aggregate,),
    )
    ledger = DeckStepLineage.for_deck_dir(deck, workspace=workspace)
    _run(ledger, compile_step, lambda: _write(aggregate, "{}"))
    assert ledger.should_skip(compile_step, resume=True).skip

    # A sidecar outside the pattern is not an input.
    _write(spec_dir / "sl_07-figure.critic.json", '{"note": "changed"}')
    assert ledger.should_skip(compile_step, resume=True).skip

    spec.write_text('{"label": "Prediction alone"}', encoding="utf-8")
    decision = ledger.should_skip(compile_step, resume=True)
    assert not decision.skip
    assert decision.reason.startswith("input_changed:") and decision.reason.endswith("figure_specs")
