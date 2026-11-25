"""v6.1 Phase 2 review round 1 (notes/reviews/phase-2-r1) regressions, one per finding."""

from __future__ import annotations

import json

import pytest

from conference_pipeline.ontology import catalog as cat
from conference_pipeline.ontology import leakage as lk
from conference_pipeline.ontology import splits as sp
from conference_pipeline.ontology import work_families as wf
from tests._ontology_fixtures import TALKS_REL, TITLE_C, aid, build_workspace, write_json


def _inventory(workspace):
    path = workspace / "storage/corpus/corpus_inventory.json"
    return path, json.loads(path.read_text())


def _family_of(roster, artifact):
    return next(f for f in roster["families"] if artifact in f["artifacts"])


# F1: a companion whose title names two stable papers in one group joins that group
def test_an_ambiguous_companion_shares_the_split_group_of_every_paper_its_title_names(tmp_path):
    workspace = build_workspace(tmp_path)
    path, inv = _inventory(workspace)
    inv["artifact_records"].append({"artifact_id": aid("paper_text:p09"), "source_family_id": "paper_text",
                                    "indexable_text": True, "material_type": "slides", "item_kind": "poster",
                                    "item_id": "poster-109", "paper_title": TITLE_C, "metadata": {}})
    write_json(path, inv)
    roster = wf.build_roster(workspace, run_id="t")
    poster = _family_of(roster, aid("paper_text:p09"))
    main = _family_of(roster, aid("paper_text:c01"))
    assert poster["split_group"] == main["split_group"]


def test_a_recording_talk_naming_two_papers_joins_both_groups(tmp_path):
    workspace = build_workspace(tmp_path)
    write_json(workspace / TALKS_REL, {"talks": [{"talk_title": TITLE_C}]})
    roster = wf.build_roster(workspace, run_id="t")
    session = _family_of(roster, aid("speech_transcript:s03"))
    assert session["split_group"] == _family_of(roster, aid("workshop_paper_text:w02"))["split_group"]
    assert session["split_group"] == _family_of(roster, aid("paper_text:c01"))["split_group"]


# F2: the receipt's title joins companions when the inventory record has none
def test_a_paper_record_without_a_title_uses_its_receipt_title_for_companion_joins(tmp_path):
    workspace = build_workspace(tmp_path)
    path, inv = _inventory(workspace)
    for record in inv["artifact_records"]:
        if isinstance(record, dict) and record.get("artifact_id") == aid("paper_text:a01"):
            record.pop("paper_title")
            record["metadata"].pop("paper_title")
    write_json(path, inv)
    roster = wf.build_roster(workspace, run_id="t")
    assert _family_of(roster, aid("paper_text:p01"))["family_key"] == "openreview:FORUMAAAA1"


# F3: the citation manifest attributes its sources to the product it names, bare ids included
def test_manifest_sources_are_attributed_to_their_product_including_bare_forum_ids(tmp_path):
    workspace = build_workspace(tmp_path)
    write_json(workspace / "analysis/analysis_citation_manifest.json", {"topics": [
        {"deep_analysis_path": "analysis/01_alpha.md", "matched_paper_ids": ["FORUMEEEE5"],
         "extraction_record_ids": ["paper_abstract:FORUMDDDD4"]}]})
    roster = wf.build_roster(workspace, run_id="t")
    rows = lk.scan_products(workspace / "analysis", wf.artifact_index(roster))
    alpha = next(r for r in rows if r["product"] == "01_alpha.md")
    assert {"openreview:FORUMEEEE5", "openreview:FORUMDDDD4"} <= set(alpha["cited_families"])
    assert "openreview:FORUMEEEE5" in lk.cited_set(rows)


# F4: a malformed citation is counted as unresolved, not dropped
def test_malformed_citation_tokens_are_counted():
    families, unresolved = lk.resolve(lk.citations("paper_text:xyz paper_abstract:REAL"), {})
    assert not families and set(unresolved) == {"paper_text:xyz", "openreview:REAL"}


# F5: malformed talk-map collections never raise
@pytest.mark.parametrize("payload", [{"talks": 42}, {"program_hints": {"scheduled_talks": True}},
                                     {"program_hints": [1]}, {"talks": [None, 3, {"talk_title": 7}]}])
def test_a_malformed_talk_map_is_counted_not_raised(tmp_path, payload):
    workspace = build_workspace(tmp_path)
    write_json(workspace / TALKS_REL, payload)
    roster = wf.build_roster(workspace, run_id="t")
    assert _family_of(roster, aid("speech_transcript:s03"))["linked_families"] == []


# F6: GND's 32 MiB limit on a model's whole reference set
def test_the_combined_reference_bytes_limit_forces_sharding(tmp_path, monkeypatch):
    roster = wf.build_roster(build_workspace(tmp_path / "ws"), run_id="t")
    one = cat.build_catalog(roster, tmp_path / "a")
    total = sum(p["bytes"] for p in one["shards"][0]["populations"].values())
    monkeypatch.setattr(cat, "GND_REFERENCE_SET_MAX_BYTES", total)
    index = cat.build_catalog(roster, tmp_path / "b")
    assert index["shard_count"] > 1
    assert all(sum(p["bytes"] for p in s["populations"].values()) <= total * cat.SHARD_HEADROOM
               for s in index["shards"])


# F7: "belongs to exactly one work" / "is in exactly one version" are mandatory as well as unique
def test_version_and_passage_ownership_are_mandatory():
    from importlib import resources

    import yaml
    seed = yaml.safe_load((resources.files("conference_pipeline.ontology") / "vocabulary" / "seed-v1.orm.yaml")
                          .read_text(encoding="utf-8"))
    by_id = {f["id"]: {c["type"]: c for c in f["constraints"]} for f in seed["fact_types"]}
    assert by_id["work_has_version"]["mandatory"]["role"] == "version"
    assert by_id["version_has_passage"]["mandatory"]["role"] == "passage"


# F8: the splits' group table must match the roster's membership
def test_split_problems_names_a_missing_or_wrong_group_table(tmp_path):
    roster = wf.build_roster(build_workspace(tmp_path), run_id="t")
    doc = sp.assign_splits(roster, frozenset(), salt="s", dev_per_topic_stratum=1, test_per_topic_stratum=1)
    assert sp.split_problems(doc, roster) == []
    emptied = {**doc, "groups": {}}
    assert any("membership" in p for p in sp.split_problems(emptied, roster))


def _gnd_missing() -> bool:
    from conference_pipeline.ontology import gnd_tool
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


@pytest.mark.skipif(_gnd_missing(), reason="gnd 0.4.0 unavailable")
def test_gnd_fails_a_version_no_work_owns(tmp_path):
    from importlib import resources

    from conference_pipeline.ontology import gnd_tool
    from conference_pipeline.ontology.gnd_tool import CommandFamily
    seed = tmp_path / "seed.orm.yaml"
    seed.write_bytes((resources.files("conference_pipeline.ontology") / "vocabulary" / "seed-v1.orm.yaml").read_bytes())
    facts = {"work_has_version": [{"work": "W1", "version": "OWNED_VERSION"}],
             "version_has_passage": [{"version": "ORPHAN_VERSION", "passage": "P1"}],
             "assertion_in_version": [{"assertion": "A1", "version": "ORPHAN_VERSION"}],
             "assertion_cites_passage": [{"assertion": "A1", "passage": "P1"}],
             "assertion_has_predicate": [{"assertion": "A1", "predicate": "improves"}],
             "assertion_has_polarity": [{"assertion": "A1", "polarity": "asserted"}],
             "assertion_attributed_to": [{"assertion": "A1", "attribution": "own"}]}
    population = tmp_path / "pop.json"
    population.write_text(json.dumps({"facts": facts, "population": {"model": seed.name, "source": "fixture",
                                                                       "covers": sorted(facts)}}))
    result = gnd_tool.run_gnd(["check", str(seed), str(population)], family=CommandFamily.CHECK, cwd=tmp_path)
    assert result.exit_code == 1 and "ORPHAN_VERSION" in result.stdout
