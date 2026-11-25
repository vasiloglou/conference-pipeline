"""v6.1 Phase 3: gold selection from the pools (spec 6.1; T6, T6b, T24)."""

from __future__ import annotations

import json

import pytest

from conference_pipeline.ontology import gold_select as gs


def _family(key, kind="main-paper", topics=(), workshops=(), group=None):
    return {"family_key": key, "kind": kind, "topics": list(topics), "workshops": list(workshops),
            "split_group": group or f"g:{key}", "artifacts": [f"a:{key}"]}


def _world():
    families = [_family(f"m{i}", topics=["01_alpha"]) for i in range(5)]
    families += [_family("w1", kind="workshop-paper", workshops=["WS/A"]),
                 _family("w2", kind="workshop-paper", workshops=["WS/B"])]
    families += [_family(f"s{i}", topics=["02_beta"]) for i in range(2)]
    roster = {"families": families, "unjoined": []}
    dev = [f"g:m{i}" for i in range(4)] + ["g:w1", "g:w2", "g:s0"]
    splits = {"splits": {"dev": dev, "sealed_test": ["g:m4", "g:s1"], "growth": []},
              "groups": {f["split_group"]: [f["family_key"]] for f in families}}
    questions = {"questions": [
        {"id": "Q-01-1", "area": "01", "priority": "core", "kind": "lookup", "evidence": '"KV cache" 10; "attention" 4'},
        {"id": "Q-01-2", "area": "01", "priority": "core", "kind": "contradiction", "evidence": '"LoRA" 3'},
        {"id": "Q-01-3", "area": "01", "priority": "core", "kind": "lookup", "evidence": '"diffusion" 5'},
        {"id": "Q-01-4", "area": "01", "priority": "extended", "kind": "lookup", "evidence": '"GAN" 2'},
        {"id": "Q-02-1", "area": "02", "priority": "core", "kind": "lookup", "evidence": '"agent" 9'},
        {"id": "Q-X-1", "area": "X", "priority": "core", "kind": "lookup", "evidence": '"benchmark" 9'},
    ]}
    texts = {"m0": "We study the KV cache.", "m1": "LoRA works.", "m2": "LoRA fails here.",
             "m3": "nothing relevant", "m4": "diffusion and kv-cache", "w1": "A diffusion model.",
             "w2": "LoRA", "s0": "an agent", "s1": "agents"}
    return splits, roster, questions, texts


def test_evidence_terms_and_whole_word_matching():
    assert gs.evidence_terms({"evidence": '"SWE-bench" 269 (main 139); "pass@k" 147'}) == ["SWE-bench", "pass@k"]
    assert gs.term_pattern("LoRA").search("we use LoRA.") and not gs.term_pattern("LoRA").search("LoRAX")
    assert gs.area_of_topic("12_social_culture") == "12"


# T6
def test_greedy_cover_with_contradictions_needing_two_works():
    splits, roster, questions, texts = _world()
    sel = gs.select_gold(splits, roster, questions, texts, per_area=3, salt="s", workshop_areas={"WS/A": "01"})
    dev01 = sel["splits"]["dev"]["01"]
    picked = [p["family"] for p in dev01["picks"]]
    assert set(picked) == {"m0", "m1", "m2"}  # KV cache, then LoRA twice for the contradiction
    assert dev01["covered"] == ["Q-01-1", "Q-01-2"] and dev01["uncovered"] == ["Q-01-3"]
    assert dev01["supplemented_from_workshops"] is False
    assert "Q-01-4" not in dev01["covered"] + dev01["uncovered"]  # extended questions do not steer


def test_ties_are_broken_by_the_salted_hash_and_the_result_is_deterministic():
    splits, roster, questions, texts = _world()
    a = gs.select_gold(splits, roster, questions, texts, per_area=4, salt="s", workshop_areas={})
    assert a == gs.select_gold(splits, roster, questions, texts, per_area=4, salt="s", workshop_areas={})
    last = a["splits"]["dev"]["01"]["picks"][-1]
    assert last["family"] == "m3" and last["new_questions"] == []


def test_a_short_area_takes_its_mapped_workshops():
    splits, roster, questions, texts = _world()
    sel = gs.select_gold(splits, roster, questions, texts, per_area=6, salt="s", workshop_areas={"WS/A": "01"})
    block = sel["splits"]["dev"]["01"]
    assert block["supplemented_from_workshops"] is True and block["candidates"] == 5
    assert "w1" in [p["family"] for p in block["picks"]] and "w2" not in [p["family"] for p in block["picks"]]
    assert "Q-01-3" in block["covered"]


# T6b
def test_selection_problems_names_an_out_of_pool_pick_and_a_double_pick():
    splits, roster, questions, texts = _world()
    sel = gs.select_gold(splits, roster, questions, texts, per_area=2, salt="s", workshop_areas={})
    assert gs.selection_problems(sel, splits) == []
    sel["splits"]["sealed_test"]["01"]["picks"].append({"family": "m0"})
    problems = gs.selection_problems(sel, splits)
    assert "m0 is picked for sealed_test but is not in its pool" in problems
    assert any(p.startswith("m0 is picked 2 times") for p in problems)


# T24
def test_feasibility_lists_core_questions_no_pick_covers():
    splits, roster, questions, texts = _world()
    sel = gs.select_gold(splits, roster, questions, texts, per_area=3, salt="s", workshop_areas={})
    report = gs.feasibility(sel, questions)
    assert report["dev"] == {"core": 4, "covered": 3, "uncovered": ["Q-01-3"]}
    assert "Q-01-2" in report["sealed_test"]["uncovered"]  # one sealed work cannot cover a contradiction


def test_an_unapproved_workshop_map_is_refused(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"schema": gs.WORKSHOP_AREAS_SCHEMA, "status": "proposed",
                                "workshops": [{"workshop": "WS/A", "area": "01"}, {"workshop": "WS/C", "area": None}]}))
    with pytest.raises(gs.SelectionInputError, match="not approved"):
        gs.load_workshop_areas(path)
    assert gs.load_workshop_areas(path, require_approved=False) == {"WS/A": "01"}


def test_load_texts_joins_a_familys_paper_texts(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t/a.txt").write_text("alpha")
    (tmp_path / "storage/corpus").mkdir(parents=True)
    (tmp_path / "storage/corpus/corpus_inventory.json").write_text(json.dumps({"artifact_records": [
        {"artifact_id": "a:m0", "material_type": "paper", "text_path": "t/a.txt"},
        {"artifact_id": "a:m1", "material_type": "paper", "text_path": "t/missing.txt"}]}))
    roster = {"families": [_family("m0"), _family("m1")]}
    assert gs.load_texts(tmp_path, roster, ["m0", "m1"]) == {"m0": "alpha", "m1": ""}
