"""v6.1 Phase 4 lane A: admission by GND's judgment route on the seed's own model (T-A1..T-A6).

Real GND 0.4.0, offline: a v5 root that includes the seed vocabulary plus one fact type of its
own (a quantity), four assertions sharing one passage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from importlib import resources

import pytest

from conference_pipeline.ontology import admission as ad
from conference_pipeline.ontology import gnd_tool


def _gnd_missing() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


pytestmark = pytest.mark.skipif(_gnd_missing(), reason=f"gnd {gnd_tool.GND_VERSION} unavailable")

SEED = resources.files("conference_pipeline.ontology") / "vocabulary" / "seed-v1.orm.yaml"
TEXT = ("GridNet reaches 0.012 error on Darcy at 64 grid points with 4 levels. GridNet does not improve on "
        "turbulent flows. Prior work reports 0.05 on Darcy. GridNet beats every baseline.\n")
QUOTES = {"a1": "GridNet reaches 0.012 error on Darcy at 64 grid points with 4 levels.",
          "a2": "GridNet does not improve on turbulent flows.", "a3": "Prior work reports 0.05 on Darcy.",
          "a4": "GridNet beats every baseline."}
A = sorted(QUOTES)
FACTS = {
    "work_has_version": [{"work": "w1", "version": "v1"}],
    "version_has_passage": [{"version": "v1", "passage": "p01"}],
    "assertion_in_version": [{"assertion": a, "version": "v1"} for a in A],
    "assertion_cites_passage": [{"assertion": a, "passage": "p01"} for a in A],  # one passage, four claims
    "assertion_has_predicate": [{"assertion": "a1", "predicate": "reaches"},
                                {"assertion": "a2", "predicate": "does not improve"},
                                {"assertion": "a3", "predicate": "reports"}, {"assertion": "a4", "predicate": "beats"}],
    "assertion_has_participant_in_role": [
        {"assertion": "a1", "participant": "GridNet", "role": "method"},
        {"assertion": "a1", "participant": "Darcy", "role": "benchmark"},
        {"assertion": "a1", "participant": "error", "role": "metric"},
        {"assertion": "a2", "participant": "GridNet", "role": "method"},
        {"assertion": "a3", "participant": "Prior work", "role": "method"},
        {"assertion": "a4", "participant": "GridNet", "role": "method"}],
    "assertion_under_condition": [{"assertion": "a1", "condition": "64 grid points"},
                                  {"assertion": "a1", "condition": "4 levels"}],
    "assertion_in_experiment": [{"assertion": "a1", "experiment": "exp-darcy"}],
    "assertion_has_polarity": [{"assertion": a, "polarity": "negated" if a == "a2" else "asserted"} for a in A],
    "assertion_attributed_to": [{"assertion": a, "attribution": "cited" if a == "a3" else "own"} for a in A],
    "assertion_reports_quantity": [{"assertion": "a1", "quantity": "0.012"}],
}


def _dump(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "run"
    (root / "vocabulary").mkdir(parents=True)
    seed = SEED.read_bytes()
    (root / "vocabulary/seed.orm.yaml").write_bytes(seed)
    (root / "checking.orm.json").write_bytes(_dump({
        "version": 5, "model_id": "urn:test:checking@1",
        "source": {"confidence": "standard", "kind": "text-conversation", "mode": "one-shot"},
        "includes": [{"id": "seed", "path": "vocabulary/seed.orm.yaml", "sha256": hashlib.sha256(seed).hexdigest()}],
        "object_types": [{"id": "quantity_value", "kind": "value", "name": "QuantityValue", "primitive": "String"}],
        "fact_types": [{"id": "assertion_reports_quantity", "reading": "{Assertion} reports {QuantityValue}",
                        "roles": [{"role_name": "assertion", "object_type": "assertion"},
                                  {"role_name": "quantity", "object_type": "quantity_value"}]}]}))
    raw = root / "population.json"
    raw.write_bytes(_dump({"population": {"model": "checking.orm.json", "model_id": "urn:test:checking@1",
                                          "source": "agent-output", "covers": sorted(FACTS)}, "facts": FACTS}))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/paper.txt").write_text(TEXT, encoding="utf-8")
    return root, raw, tmp_path / "docs"


def _verdicts(**override):
    out = {}
    for a, q in QUOTES.items():
        start = TEXT.index(q)
        out[a] = {"verdict": override.get(a, "supported"), "path": "paper.txt",
                  **ad.byte_span(TEXT, {"start": start, "end": start + len(q)})}
    return out


def test_t_a1_judgments_check_runs_on_an_include_root_through_gnd_flattening(case, tmp_path):
    root, raw, docs = case
    record = ad.admit(root / "checking.orm.json", raw, _verdicts(a4="unsupported"), docs, tmp_path / "out")
    assert record["gnd"] == {"exit": 1, "meaning": "alethic-failure"}  # not every target admitted (a4)
    assert record["schema"] == "admission-record/1" and len(record["flat_model_sha256"]) == 64


def test_t_a2_rows_without_an_assertion_role_pass_unchanged(case, tmp_path):
    root, raw, docs = case
    ad.admit(root / "checking.orm.json", raw, _verdicts(), docs, tmp_path / "out")
    out = json.loads((tmp_path / "out/admitted.population.json").read_bytes())
    for ft in ("work_has_version", "version_has_passage"):
        assert out["facts"][ft] == FACTS[ft]


def test_t_a3_composition_refuses_a_source_whose_bytes_are_not_the_admitted_ones(case, tmp_path):
    root, raw, docs = case
    ad.admit(root / "checking.orm.json", raw, _verdicts(), docs, tmp_path / "out")
    assert ad.composition_refusal({"paper": tmp_path / "out"}) is None
    (tmp_path / "out/admitted.population.json").write_bytes(b"{}")
    assert ad.composition_refusal({"paper": tmp_path / "out"}) == "composition:admitted-digest-mismatch:paper"


def test_t_a4_supported_negated_and_cited_claims_are_kept_flagged_and_counted(case, tmp_path):
    root, raw, docs = case
    record = ad.admit(root / "checking.orm.json", raw, _verdicts(a4="unsupported"), docs, tmp_path / "out")
    by = {d["assertion"]: d for d in record["assertions"]}
    assert by["a2"]["disposition"] == "supported-not-admitted" and by["a2"]["stages"]["positive-own-findings"] == "negated"
    assert by["a3"]["disposition"] == "supported-not-admitted" and by["a3"]["stages"]["positive-own-findings"] == "cited"
    assert by["a4"] == {"assertion": "a4", "index": 3, "disposition": "rejected",
                        "stages": {"semantic-support": "rejected", "positive-own-findings": None}}
    assert record["counts"] == {"admitted": 1, "supported-not-admitted": 2, "rejected": 1}
    out = json.loads((tmp_path / "out/admitted.population.json").read_bytes())
    assert {r["assertion"] for r in out["facts"]["assertion_has_predicate"]} == {"a1", "a2", "a3"}


def test_t_a5_a_changed_evidence_file_is_a_gnd_refusal(case, tmp_path, monkeypatch):
    root, raw, docs = case
    real = ad.gnd_tool.run_gnd

    def edit_then_run(args, **kw):  # the evidence changes after its receipts were written
        if args[:2] == ["judgments", "check"]:
            (docs / "paper.txt").write_text(TEXT.replace("0.012", "0.013"), encoding="utf-8")
        return real(args, **kw)
    monkeypatch.setattr(ad.gnd_tool, "run_gnd", edit_then_run)
    with pytest.raises(ad.AdmissionError, match="judgments:refused"):
        ad.admit(root / "checking.orm.json", raw, _verdicts(), docs, tmp_path / "out")


def test_t_a6_a_claim_the_flat_example_shape_cannot_hold_keeps_every_row(case, tmp_path):
    root, raw, docs = case
    record = ad.admit(root / "checking.orm.json", raw, _verdicts(), docs, tmp_path / "out")
    assert {d["assertion"]: d["disposition"] for d in record["assertions"]}["a1"] == "admitted"
    out = json.loads((tmp_path / "out/admitted.population.json").read_bytes())
    a1 = {ft: [r for r in rows if r.get("assertion") == "a1"] for ft, rows in out["facts"].items()}
    assert len(a1["assertion_has_participant_in_role"]) == 3          # three participants
    assert len(a1["assertion_under_condition"]) == 2                  # two conditions
    assert a1["assertion_reports_quantity"] == [{"assertion": "a1", "quantity": "0.012"}]  # a quantity
    assert len(out["facts"]["assertion_cites_passage"]) == 4          # one passage, four kept claims


def test_a_missing_verdict_is_missing_and_roots_gnd_cannot_flatten_are_refused(case, tmp_path):
    root, raw, docs = case
    verdicts = _verdicts()
    del verdicts["a1"]
    record = ad.admit(root / "checking.orm.json", raw, verdicts, docs, tmp_path / "out")
    assert record["assertions"][0]["stages"]["semantic-support"] == "missing"
    model = json.loads((root / "checking.orm.json").read_bytes())
    model["reference_populations"] = [{"id": "x", "path": "x.json", "sha256": "0" * 64, "fact_types": []}]
    (root / "refs.orm.json").write_bytes(_dump(model))
    with pytest.raises(ad.AdmissionError, match="admission:flatten:"):  # its catalog file is absent
        ad.admit(root / "refs.orm.json", raw, verdicts, docs, tmp_path / "out2")
    shutil.rmtree(tmp_path / "out")


def test_review_f3_a_catalog_bound_root_is_admitted_with_its_reference_identity(tmp_path):
    """Phase 4 review F3: lane X rebinds onto a root that binds a catalog; lane A must judge that root.

    GND's interop checking root, less its temporal rule (``gnd extract`` refuses temporal models,
    IF9), with the rebound replication population: supported -> GND pass; unsupported -> rejected.
    """
    from tests import _ontology_assurance_fixtures as fx

    run = fx.interop_run(tmp_path / "interop")
    model = json.loads((run / "checking.orm.json").read_bytes())
    model["constraints"] = [c for c in model["constraints"] if c["type"] != "guarded"]
    assert model["reference_populations"]
    (run / "atemporal.orm.json").write_bytes(_dump(model))
    folder = run / "sources/replication"
    start = (folder / "document.txt").read_bytes().find(b"PINNs")
    shape = ad.AssertionShape(entity_role="work", target="paper_names_parent_method", polarity=None, attribution=None)
    for verdict, meaning, disposition in (("supported", "pass", "admitted"),
                                          ("unsupported", "alethic-failure", "rejected")):
        record = ad.admit(run / "atemporal.orm.json", folder / "population.json",
                          {"example:replication-2019": {"verdict": verdict, "path": "document.txt", "start": start,
                                                        "end": start + 5}},
                          folder, tmp_path / f"out-{verdict}", shape=shape)
        assert record["gnd"]["meaning"] == meaning
        assert record["assertions"][0]["disposition"] == disposition


def test_review_r2_f6_a_root_without_includes_is_judged_in_place(tmp_path):
    from tests import _ontology_assurance_fixtures as fx

    run = fx.interop_run(tmp_path / "interop")
    flat_dir = tmp_path / "flat"
    model = json.loads((run / "checking.orm.json").read_bytes())
    model["constraints"] = [c for c in model["constraints"] if c["type"] != "guarded"]
    (run / "atemporal.orm.json").write_bytes(_dump(model))
    assert gnd_tool.run_gnd(["extract", str(run / "atemporal.orm.json"), "--out", str(flat_dir)],
                            family=gnd_tool.CommandFamily.OTHER, cwd=run).exit_code == 0
    flat = json.loads((flat_dir / "atemporal.orm.flat.orm.json").read_bytes())
    assert not flat.get("includes") and flat["reference_populations"]
    (run / "already-flat.orm.json").write_bytes(_dump(flat))
    folder = run / "sources/replication"
    start = (folder / "document.txt").read_bytes().find(b"PINNs")
    shape = ad.AssertionShape(entity_role="work", target="paper_names_parent_method", polarity=None, attribution=None)
    record = ad.admit(run / "already-flat.orm.json", folder / "population.json",
                      {"example:replication-2019": {"verdict": "supported", "path": "document.txt", "start": start,
                                                    "end": start + 5}}, folder, tmp_path / "out", shape=shape)
    assert record["gnd"]["meaning"] == "pass" and record["flattening_sha256"] is None
