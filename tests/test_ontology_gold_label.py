"""v6.1 Phase 3: the labelling host and owner adjudication (spec 6.3-6.4; T7-T10)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from conference_pipeline.ontology import gold
from conference_pipeline.ontology import gold_adjudicate as adj
from conference_pipeline.ontology import gold_label as gl

PAPER = """Fast Forking for Diffusion Language Models

Abstract
We propose UnMaskFork, a test-time search over unmasking paths for masked diffusion language models.

1 Introduction
Our main contributions are: a tree search over unmasking orders, and a study of its cost.

4 Experiments
UnMaskFork reaches 0.742 pass@1 on HumanEval, beating beam search under the same compute budget.

6 Limitations
We test on code only; mathematical reasoning is left to future work.
"""
WORK = "openreview:W1"


def _extract(role, *, extra=None):
    statements = [{"passage": "p03", "quote": "UnMaskFork reaches 0.742 pass@1 on HumanEval",
                   "predicate": "reaches", "participants": [{"role": "subject", "name": "UnMaskFork"},
                                                            {"role": "benchmark", "name": "HumanEval"}],
                   "conditions": ["same compute budget"], "experiment": "main results",
                   "quantity": {"coefficient": 742, "scale": 3, "unit": "pass@1"},
                   "polarity": "asserted", "attribution": "own"}]
    distractors = [{"passage": "p04", "quote": "We test on code only", "predicate": "generalizes to",
                    "participants": [{"role": "subject", "name": "UnMaskFork"}, {"role": "object", "name": "math"}],
                    "polarity": "asserted", "attribution": "own", "misreading": "contradicted", "why": "x"}]
    return json.dumps({"statements": statements + (extra or []), "distractors": distractors})


def _save(root, role, kind, text, stop="end"):
    path = gl.response_path(root, role, kind, WORK)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text, "stop_reason": stop}))


def _saved(tmp_path, *, check_primary=None, check_secondary=None):
    root = tmp_path / "responses"
    _save(root, "primary", "extract", _extract("primary"))
    _save(root, "secondary", "extract", "```json\n" + _extract("secondary") + "\n```")
    _save(root, "primary", "check", json.dumps({"labels": check_primary or [
        {"id": "ss01", "support": "supported"}, {"id": "sd01", "support": "contradicted"}]}))
    _save(root, "secondary", "check", json.dumps({"labels": check_secondary or [
        {"id": "ps01", "support": "supported"}, {"id": "pd01", "support": "stretched"}]}))
    return root


def test_passages_follow_the_headings_and_do_not_overlap():
    items = gl.passages(PAPER)
    assert [p["section"] for p in items] == ["abstract", "contributions", "results", "limitations"]
    for a, b in zip(items, items[1:], strict=False):
        assert a["span"]["end"] <= b["span"]["start"]


def test_a_text_without_headings_falls_back_to_windows():
    items = gl.passages("x " * 10000)
    assert {p["section"] for p in items} == {"window"} and len(items) == 2


def test_quotes_are_located_exactly_or_across_whitespace():
    span = {"start": 0, "end": len(PAPER)}
    exact = gl.locate_quote(PAPER, span, "beating beam search")
    assert PAPER[exact["start"]:exact["end"]] == "beating beam search"
    loose = gl.locate_quote(PAPER, span, "UnMaskFork   reaches\n0.742")
    assert PAPER[loose["start"]:loose["end"]] == "UnMaskFork reaches 0.742"
    assert gl.locate_quote(PAPER, span, "not in the paper at all") is None
    assert gl.locate_quote(PAPER, span, "short") is None


# T7
def test_saved_replay_makes_no_call_and_is_byte_identical(tmp_path):
    root = _saved(tmp_path)
    first = gl.label_work(WORK, PAPER, gl.SavedProducer(root))
    producer = gl.SavedProducer(root)
    second = gl.label_work(WORK, PAPER, producer)
    assert producer.calls == 0 and json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [(i["id"], i["author_support"], i["checker_support"]) for i in first["items"]] == [
        ("ps01", "supported", "supported"), ("pd01", "contradicted", "stretched"),
        ("ss01", "supported", "supported"), ("sd01", "contradicted", "contradicted")]
    span = first["items"][0]["span"]
    assert PAPER[span["start"]:span["end"]] == "UnMaskFork reaches 0.742 pass@1 on HumanEval"


# T8
@pytest.mark.parametrize("text,stop,outcome", [
    ("", "end", "empty"), ("I cannot help", "refusal", "refused"), ("not json", "end", "unparsable"),
    ('{"statements": []', "max_tokens", "truncated"),
])
def test_refused_empty_unparsable_and_truncated_responses_are_outcomes(tmp_path, text, stop, outcome):
    root = _saved(tmp_path)
    _save(root, "primary", "extract", text, stop=stop)
    labels = gl.label_work(WORK, PAPER, gl.SavedProducer(root))
    assert {"role": "primary", "item": "*", "outcome": outcome} in labels["outcomes"]
    assert not [i for i in labels["items"] if i["author"] == "primary"]


def test_a_bad_quote_and_an_invalid_item_are_outcomes_not_statements(tmp_path):
    root = _saved(tmp_path)
    bad = [{"passage": "p03", "quote": "a sentence the paper never says", "predicate": "x",
            "participants": [{"role": "s", "name": "n"}], "polarity": "asserted", "attribution": "own"},
           {"passage": "p03", "quote": "beating beam search", "predicate": "beats",
            "participants": [{"role": "s", "name": "n"}], "polarity": "maybe", "attribution": "own"},
           {"passage": "p99", "quote": "beating beam search", "predicate": "beats",
            "participants": [{"role": "s", "name": "n"}], "polarity": "asserted", "attribution": "own"}]
    _save(root, "primary", "extract", _extract("primary", extra=bad))
    outcomes = {o["item"]: o["outcome"] for o in gl.label_work(WORK, PAPER, gl.SavedProducer(root))["outcomes"]}
    assert outcomes["ps02"] == "bad-span" and outcomes["ps03"].startswith("invalid: bad polarity")
    assert outcomes["ps04"] == "unknown-passage"


# T9
def test_the_cross_check_is_blind_to_the_authors_label():
    items = gl.passages(PAPER)
    claim = {"id": "ps01", "passage": "p03", "predicate": "reaches", "participants": [{"role": "s", "name": "U"}],
             "polarity": "asserted", "author_support": "supported", "kind": "distractor", "misreading": "stretched"}
    prompt = gl.check_prompt(PAPER, items, [claim])
    assert "author_support" not in prompt and "misreading" not in prompt and "distractor" not in prompt


def test_an_unlabelled_claim_is_an_outcome(tmp_path):
    root = _saved(tmp_path, check_primary=[{"id": "ss01", "support": "supported"}])
    labels = gl.label_work(WORK, PAPER, gl.SavedProducer(root))
    assert {"role": "primary", "item": "sd01", "outcome": "unlabelled"} in labels["outcomes"]
    assert next(i for i in labels["items"] if i["id"] == "sd01")["checker_support"] is None


def test_the_live_producer_saves_responses_and_halts_at_its_cap(tmp_path):
    calls = []

    def fake_complete(task, **kwargs):
        calls.append(task)
        return SimpleNamespace(text=_extract("x"), stop_reason="end", provider="gemini", model="m",
                               input_tokens=1000, output_tokens=100, receipt={"id": len(calls)})

    # each call costs 0.01 and could cost 0.01: two fit under 0.025, a third could cross it (review r1 F9)
    producer = gl.LiveProducer(save_root=tmp_path / "saved", budget_cap_usd=0.025, complete=fake_complete,
                               estimate=lambda model, i, o, provider=None: 0.01)
    producer("primary", "extract", WORK, "p")
    assert calls == ["ontology.gold_label_primary"] and producer.spent_usd == pytest.approx(0.01)
    assert json.loads(gl.response_path(tmp_path / "saved", "primary", "extract", WORK).read_text())["cost_usd"] == 0.01
    producer("secondary", "extract", WORK, "p")
    assert calls[-1] == "ontology.gold_label_secondary"
    with pytest.raises(gl.BudgetExceededError):
        producer("primary", "check", WORK, "p")
    replay = gl.SavedProducer(tmp_path / "saved")("primary", "extract", WORK, "p")
    assert replay.text == _extract("x") and replay.provider == "gemini"


# T10
def _labels(tmp_path):
    return gl.label_work(WORK, PAPER, gl.SavedProducer(_saved(tmp_path)))


def test_the_queue_is_every_disagreement_plus_a_sample_of_agreements(tmp_path):
    labels = _labels(tmp_path)
    queued = adj.queue([labels], salt="s")
    assert [q["key"] for q in queued if q["reason"] == "disagreement"] == [f"{WORK}#pd01"]
    many = {"work": "w", "items": [{"id": f"i{n}", "author_support": "supported", "checker_support": "supported"}
                                   for n in range(2000)]}
    sampled = len(adj.queue([many], salt="s"))
    assert 150 < sampled < 250  # about 10%


def test_an_unruled_queued_item_blocks_the_gold_file(tmp_path):
    labels = _labels(tmp_path)
    with pytest.raises(adj.UnruledItemsError, match="unruled"):
        adj.gold_for_work(labels, {}, salt="s", split="dev", document={"artifact": "a"})


def test_rulings_make_a_valid_gold_file_without_duplicates(tmp_path):
    labels = _labels(tmp_path)
    keys = [q["key"] for q in adj.queue([labels], salt="s")]
    rulings = {k: {"ruling": "accept-checker", "date": "2026-11-18", "words": "stretched, not contradicted"}
               for k in keys}
    assert adj.ruling_problems(rulings) == []
    # pd01 is now stretched while the identical sd01 is contradicted: the owner must drop one
    with pytest.raises(adj.UnruledItemsError, match="state the same claim"):
        adj.gold_for_work(labels, rulings, salt="s", split="dev", document={"artifact": "a"})
    rulings[f"{WORK}#pd01"] = {"ruling": "drop", "date": "2026-11-18", "words": "duplicate of sd01"}
    doc = adj.gold_for_work(labels, rulings, salt="s", split="dev", document={"artifact": "a"})
    assert gold.gold_problems(doc, PAPER) == []
    assert sorted(s["support"] for s in doc["statements"]) == ["contradicted", "supported"]  # ps01 = ss01 merged
    assert {s["source_item"] for s in doc["statements"]} == {f"{WORK}#ps01", f"{WORK}#sd01"}


def test_drop_and_edit_rulings(tmp_path):
    labels = _labels(tmp_path)
    keys = [q["key"] for q in adj.queue([labels], salt="s")]
    rulings = {keys[0]: {"ruling": "edit", "support": "insufficient-evidence", "date": "d", "words": "w",
                         "fields": {"predicate": "transfers to"}}}
    rulings.update({k: {"ruling": "drop", "date": "d", "words": "w"} for k in keys[1:]})
    doc = adj.gold_for_work(labels, rulings, salt="s", split="dev", document={})
    edited = next(s for s in doc["statements"] if s["source_item"] == keys[0])
    assert (edited["support"], edited["predicate"]) == ("insufficient-evidence", "transfers to")
    assert adj.ruling_problems({"k": {"ruling": "maybe"}}) == ["k: unknown ruling 'maybe'",
                                                               "k: a ruling needs its date and the owner's words"]


def test_the_agreement_report_counts_by_author(tmp_path):
    report = adj.agreement_report([_labels(tmp_path)])
    assert report["by_author"] == {"primary": {"items": 2, "agreed": 1}, "secondary": {"items": 2, "agreed": 2}}


# Canary findings (2026-11-18, ICML 2026 live canary of 2 works)
def test_quotes_match_across_pdf_line_breaks_and_break_hyphens():
    text = "Existing methods attempt to ad-\ndress this issue; we propose\nPlan-Then-Action Enhanced Reason-\ning."
    span = {"start": 0, "end": len(text)}
    for quote in ("attempt to ad-dress this issue", "we proposePlan-Then-Action", "attempt to address this issue",
                  "Enhanced Reasoning."):
        found = gl.locate_quote(text, span, quote)
        assert found is not None, quote
    found = gl.locate_quote(text, span, "attempt to address this issue")
    assert text[found["start"]:found["end"]] == "attempt to ad-\ndress this issue"
    assert gl.locate_quote(text, span, "attempt to repair this issue") is None


def test_the_extract_prompt_defines_scale_as_decimal_places():
    prompt = gl.extract_prompt(PAPER, gl.passages(PAPER))
    assert "the number of decimal places, 0 or more" in prompt and "48.8% is coefficient 488, scale 1" in prompt


def test_a_response_saved_for_another_prompt_is_superseded_not_reused(tmp_path):
    calls = []

    def fake(task, **kwargs):
        calls.append(kwargs["prompt"])
        return SimpleNamespace(text="{}", stop_reason="end", provider="gemini", model="m", input_tokens=1,
                               output_tokens=1, receipt=None)

    def producer():
        return gl.LiveProducer(save_root=tmp_path, budget_cap_usd=1.0, complete=fake,
                               estimate=lambda model, i, o, provider=None: 0.01, resolve=lambda t: ("gemini", "m"))

    producer()("primary", "extract", WORK, "prompt one")
    producer()("primary", "extract", WORK, "prompt one")
    assert calls == ["prompt one"]
    again = producer()
    again("primary", "extract", WORK, "prompt two")
    assert calls == ["prompt one", "prompt two"]
    assert len(list(tmp_path.rglob("*.superseded-*.json"))) == 1
    assert producer().spent_usd == pytest.approx(0.02)  # the superseded call's cost still counts


# Owner ruling 2026-11-18 (spec §18): a setup passage, results that survive subsection headings,
# and a numeric fallback window
NESTED = """Abstract
We study a thing.

3. Experiments
3.1. Experimental Setup
We train on ImageNet with ResNet-50 baselines for 90 epochs.
Results show that this sentence is body text, not a heading.
3.2. Experimental Results
Our method reaches 81.2% top-1 accuracy (Table 2), beating the baseline by 1.4 points.

4. Conclusion
It works.
"""


def test_a_subsection_heading_does_not_end_its_section_and_setup_is_a_passage():
    items = {p["section"]: NESTED[p["span"]["start"]:p["span"]["end"]] for p in gl.passages(NESTED)}
    assert items["setup"].startswith("3.1. Experimental Setup") and "ImageNet" in items["setup"]
    assert items["results"].startswith("3.2. Experimental Results") and "81.2%" in items["results"]
    assert "4. Conclusion" not in items["results"]


def test_a_body_line_starting_with_results_is_not_a_heading():
    text = ("Abstract\nShort.\n\n2 Method\nWe describe the method here in a few words.\n"
            "Results show that the method is fast and accurate on every task we tried, by a wide margin.\n")
    assert "results" not in {p["section"] for p in gl.passages(text)}


def test_without_a_results_heading_the_densest_numeric_window_is_used():
    filler = "Some words without numbers. " * 400
    numbers = "We get 81.2% and 77.4% in Table 3, and 0.91 on Figure 2. " * 40
    text = "Abstract\nShort abstract here.\n\n" + filler + numbers + filler
    window = next(p for p in gl.passages(text) if p["section"] == "results-window")
    assert "81.2%" in text[window["span"]["start"]:window["span"]["end"]]


def test_the_primary_labeller_thinks_low_and_options_bind_the_reuse_key(tmp_path):
    seen = []

    def fake(task, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(text="{}", stop_reason="end", provider="gemini", model="m", input_tokens=1,
                               output_tokens=1, receipt=None)

    def producer(**extra):
        return gl.LiveProducer(save_root=tmp_path, budget_cap_usd=1.0, complete=fake, resolve=lambda t: ("g", "m"),
                               estimate=lambda model, i, o, provider=None: 0.001, **extra)

    producer()("primary", "extract", WORK, "p")
    assert (seen[0]["reasoning_effort"], seen[0]["max_tokens"]) == ("low", 16000)
    producer()("secondary", "extract", WORK, "p")
    assert "reasoning_effort" not in seen[1] and seen[1]["max_tokens"] == 8000
    producer(role_options={"primary": {"reasoning_effort": "high", "max_tokens": 16000}})(
        "primary", "extract", WORK, "p")
    assert len(seen) == 3  # other thinking settings: not the saved answer, asked again


# Owner ruling 2026-11-18 (spec §21): a unitless number is dimensionless, not an invalid item
@pytest.mark.parametrize("unit", [None, "", "missing"])
def test_a_quantity_without_a_unit_is_dimensionless_not_dropped(unit):
    quantity = {"coefficient": 74, "scale": 2}
    if unit != "missing":
        quantity["unit"] = unit
    extra = [{"passage": "p03", "quote": "UnMaskFork reaches 0.742 pass@1", "predicate": "scores",
              "participants": [{"role": "subject", "name": "UnMaskFork"}], "quantity": quantity,
              "polarity": "asserted", "attribution": "own"}]
    found, outcomes = gl.read_extraction(gl.Response(_extract("primary", extra=extra)), PAPER, gl.passages(PAPER),
                                         "primary")
    assert outcomes == [] and found[1]["quantity"] == {"coefficient": 74, "scale": 2, "unit": "dimensionless"}
    assert gl.normalize_quantity({"coefficient": 1, "scale": 0, "unit": "%"})["unit"] == "%"
    assert gl.normalize_quantity(None) is None
