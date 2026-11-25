"""v6.1 Phase 2: leakage audit of analysis products (spec 6.3, tests T11-T13)."""

from __future__ import annotations

import pytest

from conference_pipeline.ontology import leakage as lk
from conference_pipeline.ontology import splits as sp
from conference_pipeline.ontology import work_families as wf
from tests._ontology_fixtures import aid, build_workspace


@pytest.fixture
def setup(tmp_path):
    workspace = build_workspace(tmp_path)
    roster = wf.build_roster(workspace, run_id="t")
    return workspace, roster, wf.artifact_index(roster)


def _row(rows, product):
    return next(r for r in rows if r["product"] == product)


# T11
def test_artifact_ids_forum_links_forum_fields_and_unit_ids_resolve_to_families(setup):
    workspace, _, index = setup
    rows = lk.scan_products(workspace / "analysis", index)
    assert _row(rows, "01_alpha.md")["cited_families"] == ["openreview:FORUMAAAA1", "openreview:FORUMBBBB2"]
    assert _row(rows, "topic_strategy.json")["cited_families"] == ["openreview:FORUMCCCC3"]
    assert _row(rows, "deep_dive_topics/x.md")["cited_families"] == ["event:invited-1", "openreview:FORUMDDDD4"]
    assert lk.cited_set(rows) == {"openreview:FORUMAAAA1", "openreview:FORUMBBBB2", "openreview:FORUMCCCC3",
                                  "openreview:FORUMDDDD4", "event:invited-1"}


def test_only_named_analysis_products_are_scanned(setup):
    workspace, _, index = setup
    products = [r["product"] for r in lk.scan_products(workspace / "analysis", index)]
    assert "notes_not_a_product.md" not in products


def test_citation_tokens():
    text = (f"{aid('paper_text:a01')} x paper_abstract:Ab12_-Cd openreview.net/pdf?id=ZZZZZZ9 "
            '"forum": "QQQQQQ1" speech_transcript:0123456789abcdef0123:segment:00002')
    assert lk.citations(text) == [aid("paper_text:a01"), "speech_transcript:0123456789abcdef0123",
                                  "openreview:Ab12_-Cd", "openreview:ZZZZZZ9", "openreview:QQQQQQ1"]


# T12
def test_unresolvable_citations_are_counted_and_unreadable_products_reported(setup, monkeypatch):
    workspace, roster, index = setup
    rows = lk.scan_products(workspace / "analysis", index)
    assert _row(rows, "01_alpha.md")["unresolved"] == ["paper_text:ffffffffffffffffffffffff"]

    real = type(workspace).read_bytes

    def refuse(path):
        if path.name == "topic_strategy.json":
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(type(workspace), "read_bytes", refuse)
    rows = lk.scan_products(workspace / "analysis", index)
    assert _row(rows, "topic_strategy.json")["status"] == "unreadable"
    splits = sp.assign_splits(roster, frozenset(), salt="s", dev_per_topic_stratum=0, test_per_topic_stratum=0)
    report = lk.leakage_report(rows, splits)
    assert report["counts"]["unreadable"] == 1 and report["counts"]["unresolved_citations"] == 1


def test_cited_families_on_one_product(setup):
    workspace, _, index = setup
    families, unresolved = lk.cited_families(workspace / "analysis/01_alpha.md", index)
    assert families == {"openreview:FORUMAAAA1", "openreview:FORUMBBBB2"} and len(unresolved) == 1


# T13
def test_a_product_citing_a_held_out_family_is_contaminated_and_the_rest_clean(setup):
    workspace, roster, index = setup
    rows = lk.scan_products(workspace / "analysis", index)
    splits = sp.assign_splits(roster, frozenset(), salt="s", dev_per_topic_stratum=0, test_per_topic_stratum=0)
    group_a = next(f["split_group"] for f in roster["families"] if f["family_key"] == "openreview:FORUMAAAA1")
    splits["splits"]["growth"].remove(group_a)
    splits["splits"]["dev"].append(group_a)
    report = lk.leakage_report(rows, splits)
    status = {p["product"]: (p["status"], p["contaminated_by"]) for p in report["products"]}
    assert status["01_alpha.md"] == ("contaminated", ["openreview:FORUMAAAA1"])
    assert status["topic_strategy.json"] == ("clean", [])
    assert report["counts"]["contaminated"] == 1


def test_with_cited_works_excluded_every_product_is_clean(setup):
    workspace, roster, index = setup
    rows = lk.scan_products(workspace / "analysis", index)
    splits = sp.assign_splits(roster, lk.cited_set(rows), salt="s", dev_per_topic_stratum=5,
                              test_per_topic_stratum=5, dev_per_other_stratum=5, test_per_other_stratum=5,
                              other_kinds=frozenset({"workshop-paper", "event", "companion", "site-paper"}))
    report = lk.leakage_report(rows, splits)
    assert report["counts"]["contaminated"] == 0 and report["counts"]["clean"] == report["counts"]["products"]


# Phase 3 T13: gold outputs name held-out works, so they are contaminated products, never growth inputs
def test_gold_outputs_scanned_as_products_are_contaminated(setup):
    import json as _json
    workspace, roster, index = setup
    gold_dir = workspace / "analysis" / "gold-outputs"
    gold_dir.mkdir()
    (gold_dir / "w.gold.json").write_text(_json.dumps({"work": "openreview:FORUMAAAA1", "statements": [],
                                                       "document": {"artifact": aid("paper_text:a01")}}))
    rows = lk.scan_products(workspace / "analysis", index, extra_globs=("gold-outputs/*.gold.json",))
    splits = sp.assign_splits(roster, frozenset(), salt="s", dev_per_topic_stratum=0, test_per_topic_stratum=0)
    group = next(f["split_group"] for f in roster["families"] if f["family_key"] == "openreview:FORUMAAAA1")
    splits["splits"]["growth"].remove(group)
    splits["splits"]["dev"].append(group)
    report = lk.leakage_report(rows, splits)
    gold_row = next(p for p in report["products"] if p["product"] == "gold-outputs/w.gold.json")
    assert gold_row["status"] == "contaminated" and gold_row["contaminated_by"] == ["openreview:FORUMAAAA1"]
    assert "gold-outputs/w.gold.json" not in [r["product"] for r in lk.scan_products(workspace / "analysis", index)]
