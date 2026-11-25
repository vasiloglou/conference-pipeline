"""v6.1 Phase 3: cross-work link gold (spec §14 ruling "Candidates + labellers")."""

from __future__ import annotations

import json

import pytest

from conference_pipeline.ontology import gold
from conference_pipeline.ontology import gold_label as gl
from conference_pipeline.ontology import gold_links as lk


def _item(name, passage="p01", kind="statement"):
    return {"kind": kind, "passage": passage, "span": {"start": 0, "end": 5},
            "participants": [{"role": "subject", "name": name}]}


def _labels():
    return [{"split": "dev", "work": "W1", "items": [_item("VampNet"), _item("PINNs"), _item("ghost", kind="distractor")]},
            {"split": "dev", "work": "W2", "items": [_item("vampnet"), _item("PINN")]},
            {"split": "dev", "work": "W3", "items": [_item("ab")]},
            {"split": "sealed_test", "work": "S1", "items": [_item("VampNet")]}]


def test_candidates_fold_case_punctuation_and_plurals_within_a_split():
    pairs = lk.candidates(_labels())
    keys = sorted((p["key"], p["left"]["name"], p["right"]["name"]) for p in pairs)
    assert keys == [("pinn", "PINNs", "PINN"), ("vampnet", "VampNet", "vampnet")]
    assert all(p["split"] == "dev" for p in pairs)  # S1 never pairs with a dev work


def test_generic_names_and_the_pair_cap_bound_the_candidates():
    many = [{"split": "dev", "work": f"W{i}", "items": [_item("accuracy")]} for i in range(30)]
    assert lk.candidates(many) == []  # in more than 25 works: generic
    few = [{"split": "dev", "work": f"W{i}", "items": [_item("accuracy")]} for i in range(10)]
    assert len(lk.candidates(few)) == 45 and len(lk.candidates(few, max_pairs=7)) == 7


def _producer(tmp_path, primary, secondary, pairs):
    root = tmp_path / "r"
    for role, labels in (("primary", primary), ("secondary", secondary)):
        path = gl.response_path(root, role, "links", "dev#0000")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": json.dumps({"labels": [{"id": p["id"], "label": lab}
                                                                    for p, lab in zip(pairs, labels, strict=True)]})}))
    return gl.SavedProducer(root)




def test_link_lane_end_to_end(tmp_path):
    pairs = lk.candidates(_labels())
    pinn = next(p for p in pairs if p["key"] == "pinn")
    vamp = next(p for p in pairs if p["key"] == "vampnet")
    ordered = [pinn, vamp] if pairs[0] is pinn else [vamp, pinn]
    labels_by = {"pinn": ("same", "same"), "vampnet": ("same", "different")}
    producer = _producer(tmp_path, [labels_by[p["key"]][0] for p in ordered],
                         [labels_by[p["key"]][1] for p in ordered], ordered)
    result = lk.label_links("dev", pairs, {"W1": "VampNet PINNs", "W2": "vampnet PINN"}, producer)
    assert producer.calls == 0 and result["outcomes"] == []
    queued = [q["key"] for q in lk.link_queue(result, salt="s") if q["reason"] == "disagreement"]
    assert queued == [f"links:dev#{vamp['id']}"]
    with pytest.raises(lk.UnruledLinksError):
        lk.build_links(result, {}, salt="s")
    rulings = {q["key"]: {"ruling": "accept-checker", "date": "d", "words": "different things"}
               for q in lk.link_queue(result, salt="s")}
    links = lk.build_links(result, rulings, salt="s")
    labels = {p["left"]["name"]: p["label"] for p in links["identity_pairs"]}
    assert labels["VampNet"] == "different"
    assert {(n["kind"], n["left"], n["right"]) for n in links["name_pairs"]} == {("synonym", "PINNs", "PINN")}
    assert gold.links_problems(links, {"W1": "dev", "W2": "dev"}) == []


def test_a_missing_or_bad_link_response_is_an_outcome(tmp_path):
    pairs = lk.candidates(_labels())
    result = lk.label_links("dev", pairs, {}, gl.SavedProducer(tmp_path / "none"))
    assert {o["outcome"] for o in result["outcomes"]} == {"missing"}
    assert all(i["primary"] is None for i in result["items"])
    rulings = {q["key"]: {"ruling": "accept-author", "date": "d", "words": "w"} for q in lk.link_queue(result, salt="s")}
    with pytest.raises(lk.UnruledLinksError, match="no valid label"):
        lk.build_links(result, rulings, salt="s")


# Spec §23 finding (adjudicator canary): generic words are not names
def test_generic_words_are_not_link_names():
    from conference_pipeline.ontology import gold_links as gln
    for name in ("Models", "the authors", "our method", "all models", "base model", "3B model", "dataset",
                 "future work", "This work", "pretrained models", "Baseline"):
        assert gln.is_generic(name), name
    for name in ("GRPO", "Llama-3.1-8B", "CIFAR-10", "accuracy", "diffusion model", "OOD benchmark", "GPT-5",
                 "vision-language model", "batch size"):
        assert not gln.is_generic(name), name


def test_restrict_keeps_labelled_candidates_and_refuses_unlabelled_ones():
    import pytest

    from conference_pipeline.ontology import gold_links as gln
    labelled = {"split": "dev", "items": [{"id": "l1"}, {"id": "l2"}], "outcomes": [{"item": "l2", "outcome": "x"}]}
    kept = gln.restrict(labelled, [{"id": "l1", "split": "dev"}])
    assert [i["id"] for i in kept["items"]] == ["l1"] and kept["outcomes"] == []
    with pytest.raises(ValueError):
        gln.restrict(labelled, [{"id": "l3", "split": "dev"}])
