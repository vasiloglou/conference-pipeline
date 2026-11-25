"""v6.1 Phase 3: the owner's adjudication sheet (spec 6.4, D5 (a))."""

from __future__ import annotations

from conference_pipeline.ontology import gold_sheet as gs

TEXT = "Intro words. UnMaskFork reaches 0.742 pass@1 on HumanEval. More words after."
LABELS = [
    {"work": "W1", "split": "dev", "passages": [{"id": "p01", "section": "results", "span": {"start": 0, "end": 60}}],
     "items": []},
    {"work": "W2", "split": "sealed_test", "passages": [{"id": "p01", "section": "abstract",
                                                        "span": {"start": 0, "end": 76}}], "items": []},
]
ITEM = {"id": "ps01", "kind": "statement", "passage": "p01", "span": {"start": 13, "end": 57}, "predicate": "reaches",
        "participants": [{"role": "subject", "name": "UnMaskFork"}], "conditions": [], "experiment": None,
        "quantity": {"coefficient": 742, "scale": 3, "unit": "pass@1"}, "polarity": "asserted",
        "attribution": "own", "author": "primary", "checker": "secondary", "author_support": "supported",
        "checker_support": "stretched"}
LINK = {"id": "l1", "primary": "same", "secondary": "different",
        "left": {"work": "W1", "name": "UnMaskFork", "passage": "p01", "span": {"start": 13, "end": 23}},
        "right": {"work": "W1", "name": "HumanEval", "passage": "p01", "span": {"start": 48, "end": 57}}}
QUEUE = [{"key": "W1#ps01", "reason": "disagreement", "item": ITEM},
         {"key": "W2#ps01", "reason": "sample", "item": ITEM},
         {"key": "links:dev#l1", "reason": "disagreement", "item": LINK},
         {"key": "links:sealed_test#l2", "reason": "sample", "item": LINK}]


def _cards(splits=("dev",)):
    return gs.sheet_items(QUEUE, LABELS, {"W1": TEXT, "W2": TEXT}, {"W1": "Paper one"}, splits_wanted=splits)


def test_only_the_wanted_splits_reach_the_sheet():
    assert [c["key"] for c in _cards()] == ["W1#ps01", "links:dev#l1"]
    assert [c["key"] for c in _cards(("dev", "sealed_test"))] == [q["key"] for q in QUEUE]


def test_a_card_carries_both_labels_and_the_quoted_words_inside_the_passage():
    card = _cards()[0]
    assert card["quote"] == "UnMaskFork reaches 0.742 pass@1 on HumanEval"
    assert card["before"] == "Intro words. " and card["after"] == ". M"  # clipped at the passage end (60)
    assert card["labels"] == {"primary": "supported", "secondary": "stretched"} and card["title"] == "Paper one"
    link = _cards()[1]
    assert [s["quote"] for s in link["sides"]] == ["UnMaskFork", "HumanEval"]
    assert link["labels"] == {"primary": "same", "secondary": "different"}


def test_doc_ids_are_safe_and_distinct():
    ids = {gs.doc_id(q["key"]) for q in QUEUE}
    assert len(ids) == len(QUEUE) and all(set(i) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
                                          for i in ids)


def test_rows_become_rulings_and_bad_rows_are_refused():
    keys = [q["key"] for q in QUEUE]
    rows = [{"id": gs.doc_id("W1#ps01"), "key": "W1#ps01", "ruling": "edit", "support": "stretched",
             "date": "2026-11-19", "words": "edit: stretched"},
            {"id": gs.doc_id("links:dev#l1"), "key": "links:dev#l1", "ruling": "accept-checker",
             "date": "2026-11-19", "words": "accept Opus"}]
    rulings, problems = gs.rulings_from_rows(rows, keys)
    assert problems == [] and rulings["W1#ps01"]["support"] == "stretched"
    bad = [{"id": "x", "key": "nope", "ruling": "drop", "date": "d", "words": "w"},
           {"id": "wrong", "key": "W1#ps01", "ruling": "drop", "date": "d", "words": "w"},
           {"id": gs.doc_id("W2#ps01"), "key": "W2#ps01", "ruling": "edit", "date": "d", "words": "w"}]
    _, problems = gs.rulings_from_rows(bad, keys)
    assert len(problems) == 3 and "an edit names its support label" in problems[-1]
