"""v6.1 Phase 2: splits over split groups (spec 6.2, tests T7-T10b)."""

from __future__ import annotations

import json

import pytest

from conference_pipeline.ontology import splits as sp
from conference_pipeline.ontology import work_families as wf
from tests._ontology_fixtures import build_workspace


def _family(key, kind="main-paper", topics=(), group=None, workshops=()):
    return {"family_key": key, "kind": kind, "topics": list(topics), "workshops": list(workshops),
            "split_group": group or f"g:{key}", "artifacts": [f"a:{key}"]}


def _roster(families):
    return {"families": families, "unjoined": []}


def _synthetic(n_per_topic=20):
    families = []
    for topic in ("t_big", "t_small"):
        count = n_per_topic if topic == "t_big" else 6
        families += [_family(f"openreview:{topic}{i:03d}", topics=[topic]) for i in range(count)]
    # a paper listing the small topic second: rarest-topic strata put it with t_small
    families.append(_family("openreview:both", topics=["t_big", "t_small"]))
    families += [_family(f"openreview:w{i}", kind="workshop-paper", workshops=["WS/A"]) for i in range(8)]
    families += [_family(f"event:e{i}", kind="event") for i in range(3)]
    families.append(_family("openreview:none", topics=[]))
    return _roster(families)


def _assign(roster, cited=frozenset(), salt="s1", **kw):
    args = {"dev_per_topic_stratum": 3, "test_per_topic_stratum": 3, "dev_per_other_stratum": 1,
            "test_per_other_stratum": 1, **kw}
    return sp.assign_splits(roster, cited, salt=salt, **args)


def _where(doc):
    return {g: name for name in sp.SPLIT_NAMES for g in doc["splits"][name]}


# T7
def test_every_group_is_in_exactly_one_split():
    roster = _synthetic()
    doc = _assign(roster)
    assert sp.split_problems(doc, roster) == []
    assert sorted(_where(doc)) == sorted({f["split_group"] for f in roster["families"]})


def test_on_the_fixture_workspace_a_union_group_stays_together(tmp_path):
    roster = wf.build_roster(build_workspace(tmp_path), run_id="t")
    doc = _assign(roster, dev_per_other_stratum=0, test_per_other_stratum=0)
    assert sp.split_problems(doc, roster) == []
    group = next(g for g, fams in doc["groups"].items() if "event:session-900" in fams)
    assert set(doc["groups"][group]) == {"event:session-900", "openreview:FORUMDDDD4", "openreview:FORUMEEEE5"}


# T8 (D2 b)
def test_a_cited_group_is_never_in_dev_or_sealed_test():
    roster = _synthetic()
    cited = frozenset(f["family_key"] for f in roster["families"] if f["family_key"].endswith(("000", "001", "002")))
    doc = _assign(roster, cited=cited, dev_per_topic_stratum=10, test_per_topic_stratum=10)
    where = _where(doc)
    assert all(where[f"g:{key}"] == "growth" for key in cited)
    assert doc["strata"]["main-paper:t_big"]["cited"] == 3


# T9
def test_strata_get_their_pool_sizes_and_growth_keeps_half():
    doc = _assign(_synthetic())
    strata = doc["strata"]
    assert (strata["main-paper:t_big"]["dev"], strata["main-paper:t_big"]["sealed_test"]) == (3, 3)
    # t_small has 7 eligible groups: the pools take at most half (3), split 2 + 1
    assert strata["main-paper:t_small"]["groups"] == 7
    assert (strata["main-paper:t_small"]["dev"], strata["main-paper:t_small"]["sealed_test"]) == (2, 1)
    assert (strata["workshop-paper:WS/A"]["dev"], strata["workshop-paper:WS/A"]["sealed_test"]) == (1, 1)
    for untouched in ("event", "main-paper:untopiced"):
        assert strata[untouched]["dev"] == strata[untouched]["sealed_test"] == 0


@pytest.mark.parametrize("eligible,asked,expected", [
    (120, (30, 30), (30, 30)), (100, (30, 30), (25, 25)), (11, (30, 30), (3, 2)), (1, (30, 30), (0, 0)), (0, (1, 1), (0, 0)),
    (10, (0, 0), (0, 0)), (10, (8, 0), (5, 0)), (61, (30, 30), (15, 15)),
])
def test_pool_counts(eligible, asked, expected):
    assert sp.pool_counts(eligible, *asked) == expected


def test_rarest_and_first_topic_rules_differ_for_a_multi_topic_paper():
    roster = _synthetic()
    freq = sp.topic_frequencies(roster)
    both = next(f for f in roster["families"] if f["family_key"] == "openreview:both")
    assert sp.family_stratum(both, {}, rule="rarest", frequencies=freq) == "main-paper:t_small"
    assert sp.family_stratum(both, {}, rule="first", frequencies=freq) == "main-paper:t_big"
    with pytest.raises(ValueError):
        sp.family_stratum(both, {}, rule="loudest")


def test_a_workshop_takes_the_dominant_topic_of_main_papers_in_its_groups():
    roster = _roster([
        _family("openreview:m1", topics=["t1"], group="g:x"), _family("openreview:w1", kind="workshop-paper",
                                                                       workshops=["WS/B"], group="g:x"),
        _family("openreview:w2", kind="workshop-paper", workshops=["WS/C"]),
    ])
    assert sp.workshop_topics(roster) == {"WS/B": "t1"}
    assert sp.family_stratum(roster["families"][1], sp.workshop_topics(roster)) == "workshop-paper:t1"
    assert sp.family_stratum(roster["families"][2], sp.workshop_topics(roster)) == "workshop-paper:WS/C"


# T10 (R10)
def test_same_inputs_and_salt_give_identical_bytes_and_a_new_salt_changes_them():
    roster = _synthetic()
    first = wf.canonical_bytes(_assign(roster))
    assert wf.canonical_bytes(_assign(json.loads(json.dumps(roster)))) == first
    other = _assign(roster, salt="s2")
    assert wf.canonical_bytes(other) != first
    assert other["splits"]["dev"] != json.loads(first)["splits"]["dev"]


def test_the_roster_digest_ignores_the_run_id():
    roster = _synthetic()
    assert sp.roster_content_sha256({**roster, "run_id": "a"}) == sp.roster_content_sha256({**roster, "run_id": "b"})


# T10b
def test_split_problems_names_a_group_in_two_splits_and_one_in_none():
    roster = _synthetic()
    doc = _assign(roster)
    moved = doc["splits"]["growth"].pop()
    doc["splits"]["dev"].append(doc["splits"]["growth"][0])
    problems = sp.split_problems(doc, roster)
    assert f"group {moved} is in no split" in problems
    assert any(p.startswith(f"group {doc['splits']['growth'][0]} is in dev, growth") or
               p.startswith(f"group {doc['splits']['growth'][0]} is in growth, dev") for p in problems)


def test_split_problems_names_a_family_in_two_groups_a_stray_group_and_a_roster_change():
    roster = _synthetic()
    doc = _assign(roster)
    first, second = list(doc["groups"])[:2]
    doc["groups"][second] = doc["groups"][second] + doc["groups"][first]
    doc["splits"]["growth"].append("g:stray")
    roster["families"].append(_family("openreview:late"))
    problems = sp.split_problems(doc, roster)
    assert f"family {doc['groups'][first][0]} is in 2 groups" in problems
    assert "group g:stray is not in the roster" in problems
    assert "splits were made from a different roster" in problems
