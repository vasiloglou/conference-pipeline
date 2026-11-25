"""v6.1 Phase 3: the two-root assurance adapter (spec 6.7-6.8, T17-T20), corpus-first.

The expected exits are narrated in tests/_ontology_assurance_fixtures.py.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from conference_pipeline.ontology import assurance, gnd_tool
from tests import _ontology_assurance_fixtures as fx

ICML_FLOORS = Path(__file__).resolve().parents[1] / "conferences/icml2026/ontology/floors.json"


def _gnd_missing() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


pytestmark = pytest.mark.skipif(_gnd_missing(), reason=f"gnd {gnd_tool.GND_VERSION} unavailable")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("assurance")
    return fx.interop_run(root / "interop")


def _case(tmp_path, base, name, *, floors=None):
    import shutil
    run = tmp_path / "run"
    shutil.copytree(base, run)
    corpus = fx.split_corpus(tmp_path / "pair", run)
    key = fx.answers(tmp_path / "answers", run, assurance.checking_corpus_sha256(corpus, run),
                     merge_label="different" if name == "false-merge" else "same")
    if name in fx.PLANTS:
        fx.PLANTS[name](run)
    return assurance.score_two_root(corpus, key, run, floors=floors or assurance.load_floors(ICML_FLOORS))


def _stages(report):
    return {s["stage"]: s for s in report["stages"]}


# T17
def test_the_faithful_two_root_reference_run_exits_0(tmp_path, base):
    report = _case(tmp_path, base, "reference")
    assert report["exit_code"] == 0, report
    stages = _stages(report)
    for name in ("model-population-validity", "source-binding", "semantic-support", "identity-decisions",
                 "freshness"):
        assert stages[name]["status"] == "pass", (name, stages[name])
    assert stages["semantic-support"]["root"] == "extraction"
    assert stages["source-binding"]["root"] == "checking"
    assert report["floors_sha256"] == assurance.floors_sha256(assurance.load_floors(ICML_FLOORS))


# T18
@pytest.mark.parametrize("name", ["crossed-roots", "stale-rebind"])
def test_crossed_roots_and_a_stale_rebind_are_refused_naming_the_source(tmp_path, base, name):
    report = _case(tmp_path, base, name)
    code, source = fx.EXPECTED[name]
    assert report["exit_code"] == code and report["stages"] == []
    assert source in report["reason"]


# T19
@pytest.mark.parametrize("name", ["drop", "stale", "false-merge", "shifted-span"])
def test_a_planted_fault_exits_1_with_its_stage_failed(tmp_path, base, name):
    report = _case(tmp_path, base, name)
    code, failed = fx.EXPECTED[name]
    assert report["exit_code"] == code, report["failed_stages"]
    for stage in failed:
        assert stage in report["failed_stages"]


# T20
def test_floors_decide_a_stage_exactly_at_the_boundary(tmp_path, base):
    drop = _case(tmp_path / "a", base, "drop")
    recall = _stages(drop)["semantic-support"]["recall"]
    exact = Fraction(recall["numerator"], recall["denominator"])
    floors = assurance.load_floors(ICML_FLOORS)
    at = {**floors, "semantic-support": {**floors["semantic-support"], "recall": (exact.numerator, exact.denominator)}}
    report = _case(tmp_path / "b", base, "drop", floors=at)
    assert _stages(report)["semantic-support"]["status"] == "pass"
    above = Fraction(exact.numerator * 100 + 1, exact.denominator * 100)
    over = {**floors, "semantic-support": {**floors["semantic-support"], "recall": (above.numerator, above.denominator)}}
    assert _stages(_case(tmp_path / "c", base, "drop", floors=over))["semantic-support"]["status"] == "fail"


def test_the_icml_floors_file_names_every_scored_stage_and_metric():
    floors = assurance.load_floors(ICML_FLOORS)
    assert set(floors) == set(assurance.PACKAGED_FLOOR_SHAPE)
    for stage, metrics in floors.items():
        assert set(metrics) == assurance.PACKAGED_FLOOR_SHAPE[stage]
    raw = json.loads(ICML_FLOORS.read_text())
    assert raw["schema"] == "ontology-floors/1"


@pytest.mark.parametrize("edit,message", [
    (lambda d: d["stages"].pop("freshness"), "freshness"),
    (lambda d: d["stages"]["semantic-support"].update(recall=[9, 0]), "recall"),
    (lambda d: d["stages"]["semantic-support"].update(extra=[1, 2]), "extra"),
])
def test_malformed_floors_are_refused(tmp_path, edit, message):
    raw = json.loads(ICML_FLOORS.read_text())
    edit(raw)
    path = tmp_path / "floors.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(assurance.AssuranceInputError, match=message):
        assurance.load_floors(path)


def _prepared(tmp_path, base):
    import shutil
    run = tmp_path / "run"
    shutil.copytree(base, run)
    corpus = fx.split_corpus(tmp_path / "pair", run)
    key = fx.answers(tmp_path / "answers", run, assurance.checking_corpus_sha256(corpus, run))
    return run, corpus, key


@pytest.mark.parametrize("edit,reason", [
    (lambda run: (run / "sources/survey/rebind.json").unlink(), "rebind:absent:survey"),
    (lambda run: fx._rebind(run, "survey", checking_root_sha256="1" * 64), "rebind:checking-root-stale:survey"),
    (lambda run: fx._rebind(run, "survey", extraction_root_sha256="1" * 64), "rebind:extraction-root-stale:survey"),
    (lambda run: fx._rebind(run, "survey", document_sha256="1" * 64), "rebind:document-stale:survey"),
])
def test_every_binding_refusal_names_its_source(tmp_path, base, edit, reason):
    run, corpus, key = _prepared(tmp_path, base)
    edit(run)
    report = assurance.score_two_root(corpus, key, run, floors=assurance.load_floors(ICML_FLOORS))
    assert (report["exit_code"], report["reason"]) == (3, reason)


def test_an_answer_key_for_other_roots_or_of_another_schema_is_refused(tmp_path, base):
    run, corpus, key = _prepared(tmp_path, base)
    floors = assurance.load_floors(ICML_FLOORS)
    raw = json.loads(key.read_text())
    key.write_text(json.dumps({**raw, "corpus_sha256": "0" * 64}))
    assert assurance.score_two_root(corpus, key, run, floors=floors)["reason"] == "answers:corpus-mismatch"
    key.write_text(json.dumps({**raw, "schema": "other"}))
    assert assurance.score_two_root(corpus, key, run, floors=floors)["reason"] == "answers:schema"


def test_an_unparsable_adapter_output_is_a_refusal(tmp_path, base, monkeypatch):
    run, corpus, key = _prepared(tmp_path, base)

    def silent(*args, **kwargs):
        return gnd_tool.GndResult(("python",), gnd_tool.CommandFamily.OTHER, 0, "success", "not json", "", 0.0)

    monkeypatch.setattr(assurance.gnd_tool, "run_gnd_python", silent)
    report = assurance.score_two_root(corpus, key, run, floors=assurance.load_floors(ICML_FLOORS))
    assert report["exit_code"] == 3 and report["reason"].startswith("adapter:")


def test_derivations_are_found_through_includes(tmp_path, base):
    assert assurance.has_derivations(base / "checking.orm.json") is False
    model = tmp_path / "m.orm.json"
    inc = tmp_path / "inc.orm.json"
    inc.write_text(json.dumps({"fact_types": [{"id": "d", "derivation_rule": {"kind": "x"}}]}))
    model.write_text(json.dumps({"fact_types": [], "includes": [{"path": "inc.orm.json"}]}))
    assert assurance.has_derivations(model) is True


# T12: a whole dev scoring under the sealed guard reads no sealed byte
def test_dev_scoring_under_the_sealed_guard_reads_no_sealed_byte(tmp_path, base):
    from conference_pipeline.ontology import gold
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "answer-key.json").write_text("{}")
    run, corpus, key = _prepared(tmp_path, base)
    with gold.sealed_guard(sealed) as attempts:
        report = assurance.score_two_root(corpus, key, run, floors=assurance.load_floors(ICML_FLOORS))
    assert report["exit_code"] == 0 and attempts == []
