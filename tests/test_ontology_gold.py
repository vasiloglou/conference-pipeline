"""v6.1 Phase 3: gold formats and the dev / sealed split pairs (spec 6.2, 6.5; T1-T5, T11-T12)."""

from __future__ import annotations

import copy
import json

import pytest

from conference_pipeline.ontology import gold as g

TEXT = ("Abstract. We propose UnMaskFork. Results: UnMaskFork reaches 0.742 pass@1 on HumanEval, "
        "beating beam search under the same compute budget. Limitations: tested on code only.")


def _gold(work="openreview:W1", split="dev"):
    start = TEXT.index("Results")
    end = TEXT.index(" Limitations")
    claim = TEXT.index("UnMaskFork reaches")
    return {
        "schema": g.GOLD_SCHEMA, "work": work, "split": split,
        "document": {"artifact": "paper_text:x", "path": "p", "sha256": "s"},
        "passages": [{"id": "p01", "span": {"start": start, "end": end}, "section": "results"}],
        "statements": [{
            "id": "s01", "passage": "p01", "span": {"start": claim, "end": end}, "predicate": "outperforms",
            "participants": [{"role": "subject", "name": "UnMaskFork"}, {"role": "object", "name": "beam search"},
                             {"role": "benchmark", "name": "HumanEval"}, {"role": "metric", "name": "pass@1"}],
            "conditions": ["same compute budget"], "experiment": "main results",
            "quantity": {"coefficient": 742, "scale": 3, "unit": "fraction"},
            "polarity": "asserted", "attribution": "own", "support": "supported",
            "labels": {"primary": "supported", "secondary": "supported", "adjudicated": None}}],
        "coverage": [{"section": "results", "inspected": True}],
    }


# T1
def test_a_valid_gold_file_has_no_problems():
    assert g.gold_problems(_gold(), TEXT) == []


# T2
@pytest.mark.parametrize("edit,needle", [
    (lambda d: d["statements"][0].update(span={"start": 0, "end": len(TEXT) + 5}), "outside the text"),
    (lambda d: d["statements"][0].update(span={"start": 0, "end": 9}), "outside passage p01"),
    (lambda d: d["statements"].append(copy.deepcopy(d["statements"][0])), "duplicated"),
    (lambda d: d["passages"].append(copy.deepcopy(d["passages"][0])), "passage p01 is duplicated"),
    (lambda d: d["statements"][0].update(polarity="maybe"), "polarity"),
    (lambda d: d["statements"][0].update(attribution="theirs"), "attribution"),
    (lambda d: d["statements"][0].update(passage="p99"), "unknown passage"),
    (lambda d: d["statements"][0].update(participants=[]), "participants"),
    (lambda d: d["statements"][0].update(predicate=" "), "predicate"),
    (lambda d: d.update(split="growth"), "unknown split"),
])
def test_bad_spans_ids_and_enums_are_named(edit, needle):
    gold = _gold()
    edit(gold)
    assert any(needle in p for p in g.gold_problems(gold, TEXT)), g.gold_problems(gold, TEXT)


# T3
def test_quantities_are_exact_integers():
    q = _gold()["statements"][0]["quantity"]
    assert g.quantity_problems(q) == [] and g.quantity_value(q) == (742, 1000)
    assert g.quantity_problems({"coefficient": 0.742, "scale": 0, "unit": "fraction"}) == [
        "quantity coefficient must be an Integer"]
    assert "quantity scale must be a non-negative Integer" in g.quantity_problems(
        {"coefficient": 1, "scale": -1, "unit": "x"})
    assert g.quantity_problems({"coefficient": 1, "scale": 0}) == ["quantity unit must be named"]


# T4
def test_support_labels_are_exactly_gnds_four():
    assert g.SUPPORT == {"supported", "stretched", "contradicted", "insufficient-evidence"}
    gold = _gold()
    gold["statements"][0]["support"] = "partly"
    assert any("not one of GND's four" in p for p in g.gold_problems(gold, TEXT))


# T5
def test_links_stay_inside_their_split():
    works = {"openreview:W1": "dev", "openreview:W2": "dev", "openreview:S1": "sealed_test"}
    links = {"schema": g.LINKS_SCHEMA, "split": "dev",
             "identity_pairs": [{"id": "i1", "label": "same", "left": {"work": "openreview:W1", "name": "A"},
                                 "right": {"work": "openreview:W2", "name": "A"}}],
             "name_pairs": [{"id": "n1", "kind": "homonym"}],
             "chains": [{"id": "c1", "label": "supported", "works": ["openreview:W1", "openreview:W2"]}]}
    assert g.links_problems(links, works) == []
    links["identity_pairs"][0]["right"]["work"] = "openreview:S1"
    links["chains"][0]["label"] = "maybe"
    problems = g.links_problems(links, works)
    assert any("which is in sealed_test, not dev" in p for p in problems)
    assert any("chain c1 label" in p for p in problems)


def _pair(tmp_path, name, works, split):
    golds = [(_gold(w, split), TEXT.encode(), "area-1") for w in works]
    links = {"schema": g.LINKS_SCHEMA, "split": split, "identity_pairs": [], "name_pairs": [], "chains": [],
             "reuse_cases": []}
    return g.build_split_pair(tmp_path / name, split, golds, links)


# T11
def test_split_pairs_are_separate_and_pair_problems_names_a_shared_work(tmp_path):
    dev = _pair(tmp_path, "dev", ["openreview:W1"], "dev")
    sealed = _pair(tmp_path, "sealed", ["openreview:S1"], "sealed_test")
    assert dev["corpus_sha256"] != sealed["corpus_sha256"]
    corpus = json.loads((tmp_path / "dev/corpus.json").read_text())
    assert corpus["documents"][0]["id"] == "openreview-W1" and corpus["passages"][0]["id"] == "openreview-W1.p01"
    problems = g.pair_problems(tmp_path / "dev", tmp_path / "sealed")
    assert any("share document" in p for p in problems)  # the same text bytes in both
    assert not any("share work" in p for p in problems)
    _pair(tmp_path, "sealed2", ["openreview:W1"], "sealed_test")
    assert any("share work openreview:W1" in p for p in g.pair_problems(tmp_path / "dev", tmp_path / "sealed2"))


def test_a_gold_of_the_other_split_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is in sealed_test, not dev"):
        g.build_split_pair(tmp_path / "x", "dev", [(_gold("openreview:S1", "sealed_test"), b"t", "a")], {})


def test_move_sealed_returns_the_bundle_digest_and_leaves_nothing_behind(tmp_path):
    _pair(tmp_path, "sealed", ["openreview:S1"], "sealed_test")
    digest = g.move_sealed(tmp_path / "sealed", tmp_path / "elsewhere" / "sealed")
    assert len(digest) == 64 and not (tmp_path / "sealed").exists()
    assert (tmp_path / "elsewhere/sealed/corpus.json").is_file()


# T12
def test_reading_under_the_sealed_guard(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "key.json").write_text("{}")
    (tmp_path / "dev.json").write_text("{}")
    with g.sealed_guard(sealed) as attempts:
        assert (tmp_path / "dev.json").read_text() == "{}"
        with pytest.raises(g.SealedReadError):
            (sealed / "key.json").read_text()
    assert attempts == [str((sealed / "key.json").resolve())]
    assert (sealed / "key.json").read_text() == "{}"  # the guard is off after the block
