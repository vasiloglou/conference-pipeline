"""v6.1 Phase 3: mapping gold into a candidate schema (spec 6.6; T14-T16)."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology import gold_mapping as gm
from conference_pipeline.ontology.gnd_tool import CommandFamily

TEXT = "Abstract. XPINN extends PINNs with domain decomposition. It is not a transformer."
SEED = Path(str(resources.files("conference_pipeline.ontology") / "vocabulary" / "seed-v1.orm.yaml"))


def _gold():
    p = TEXT.index("XPINN")
    return {"work": "openreview:W1", "split": "dev", "passages": [{"id": "p01", "span": {"start": 0, "end": len(TEXT)}}],
            "statements": [
                {"id": "s01", "passage": "p01", "span": {"start": p, "end": TEXT.index(" It is")}, "predicate": "extends",
                 "participants": [{"role": "subject", "name": "XPINN"}, {"role": "object", "name": "PINNs"}],
                 "conditions": ["domain decomposition"], "polarity": "asserted", "attribution": "own",
                 "support": "supported"},
                {"id": "s02", "passage": "p01", "span": {"start": TEXT.index("It is"), "end": len(TEXT)},
                 "predicate": "is a", "participants": [{"role": "subject", "name": "XPINN"},
                                                        {"role": "object", "name": "transformer"}],
                 "polarity": "negated", "attribution": "own", "support": "contradicted"}]}


FACT_TYPES = {"method_builds_on_method": ["method", "base_method"]}


def _mapping():
    return {"schema": gm.MAPPING_SCHEMA, "mapper": "fixture", "statements": [
        {"work": "openreview:W1", "statement": "s01",
         "facts": [{"fact_type": "method_builds_on_method", "roles": {"method": "XPINN", "base_method": "PINNs"}}]},
        {"work": "openreview:W1", "statement": "s02",
         "facts": [{"fact_type": "method_builds_on_method", "roles": {"method": "XPINN", "base_method": "transformer"}}]}],
        "identity_pairs": []}


# T14
def test_a_complete_mapping_has_no_problems():
    assert gm.mapping_problems(_mapping(), [_gold()], FACT_TYPES) == []


def test_a_statement_can_be_unexpressible_with_a_reason():
    mapping = _mapping()
    mapping["statements"][1] = {"work": "openreview:W1", "statement": "s02", "facts": [],
                                "unexpressible": {"missing": ["polarity"], "reason": "the schema has no negation"}}
    assert gm.mapping_problems(mapping, [_gold()], FACT_TYPES) == []
    mapping["statements"][1]["unexpressible"] = {"missing": ["vibes"], "reason": ""}
    assert any("unexpressible needs" in p for p in gm.mapping_problems(mapping, [_gold()], FACT_TYPES))


def test_the_seed_can_carry_a_statement():
    fact_types = gm.model_fact_types(SEED)
    assert fact_types["assertion_has_participant_in_role"] == ["assertion", "participant", "role"]
    mapping = {"schema": gm.MAPPING_SCHEMA, "statements": [
        {"work": "openreview:W1", "statement": "s01", "facts": [
            {"fact_type": "assertion_has_predicate", "roles": {"assertion": "openreview:W1#s01", "predicate": "extends"}},
            {"fact_type": "assertion_has_participant_in_role",
             "roles": {"assertion": "openreview:W1#s01", "participant": "XPINN", "role": "subject"}}]},
        {"work": "openreview:W1", "statement": "s02", "facts": [
            {"fact_type": "assertion_has_polarity", "roles": {"assertion": "openreview:W1#s02", "polarity": "negated"}}]}]}
    assert gm.mapping_problems(mapping, [_gold()], fact_types) == []


# T15
@pytest.mark.parametrize("edit,needle", [
    (lambda m: m["statements"][0]["facts"][0]["roles"].update(base_method="ResNet"), "'ResNet' is not in the statement"),
    (lambda m: m["statements"][0]["facts"][0].update(fact_type="method_cures_cancer"), "unknown fact type"),
    (lambda m: m["statements"][0]["facts"][0]["roles"].pop("base_method"), "roles ['method'] are not"),
    (lambda m: m["statements"].pop(), "s02 is not mapped"),
    (lambda m: m["statements"].append({"work": "openreview:W1", "statement": "s99", "facts": []}), "names no gold"),
    (lambda m: m["statements"][0].update(unexpressible={"missing": ["unit"], "reason": "x"}), "not both or neither"),
])
def test_invented_participants_unknown_types_and_gaps_are_named(edit, needle):
    mapping = _mapping()
    edit(mapping)
    assert any(needle in p for p in gm.mapping_problems(mapping, [_gold()], FACT_TYPES))


def test_gold_population_and_render_manifest():
    population = gm.gold_population(_mapping(), _gold(), "model.orm.yaml")
    assert population["facts"] == {"method_builds_on_method": [{"method": "XPINN", "base_method": "PINNs"}]}
    raw = json.dumps(population).encode()
    manifest = gm.render_manifest(TEXT.encode(), raw, _mapping(), _gold(), TEXT, model_name="model.orm.yaml",
                                  model_sha256="m", applicable_sha256="a", mapper="fixture")
    assert manifest["distractors"] == [{"kind": "absent-fact", "fact_type": "method_builds_on_method",
                                        "roles": {"method": "XPINN", "base_method": "transformer"},
                                        "text": "It is not a transformer."}]


def _gnd_missing() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


# T16
@pytest.mark.skipif(_gnd_missing(), reason="gnd 0.4.0 unavailable")
def test_the_answer_key_passes_gnds_own_shape_check(tmp_path):
    links = {"identity_pairs": [{"id": "i1", "label": "same", "passages": ["W1.p01"],
                                 "left": {"work": "openreview:W1", "name": "XPINN"},
                                 "right": {"work": "openreview:W1", "name": "XPINN"}}],
             "name_pairs": [{"id": "n1", "kind": "synonym", "left": "PINNs", "right": "PINN"}]}
    mapping = {**_mapping(), "identity_pairs": [{"id": "i1", "object_type": "method"}]}
    key = gm.answer_key(mapping, [_gold()], links, {"openreview:W1": TEXT}, {"openreview:W1": "W1"},
                        corpus_sha256="0" * 64, expected_exits={"W1": 0},
                        labeller={"kind": "two-model", "independent": False})
    assert [a["support"] for a in key["assertions"]] == ["supported", "contradicted"]
    assert {a["role"] for a in key["anchored_facts"]} == {"method", "base_method"}
    assert key["identity_pairs"][0]["left"] == {"source": "W1", "object_type": "method", "value": "XPINN"}
    path = gm.write_answers(tmp_path / "answers", key, {"W1": {"facts": {}}}, {"W1": {"distractors": []}})
    script = ("import json,sys\nfrom gnd.examples.literature import assurance_report as ar\n"
              "ar.check_key_shape(json.load(open(sys.argv[1])))\nprint('ok')\n")
    result = gnd_tool.run_gnd_python(["-I", "-c", script, str(path)], family=CommandFamily.OTHER, cwd=tmp_path)
    assert result.exit_code == 0 and result.stdout.strip() == "ok", result.stderr
    assert (tmp_path / "answers/gold/W1.population.json").is_file()
