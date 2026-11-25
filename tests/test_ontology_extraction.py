"""v6.1 Phase 4 lane X: extraction bundle, per-source population, anchors and rebind (T-X1..T-X4).

Real GND 0.4.0, offline, on GND's own two-root interop example (as Phase 3's assurance tests).
"""

from __future__ import annotations

import json
import shutil

import pytest

from conference_pipeline.ontology import extraction as ex
from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology.assurance import binding_refusal
from tests import _ontology_assurance_fixtures as fx


def _gnd_missing() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


pytestmark = pytest.mark.skipif(_gnd_missing(), reason=f"gnd {gnd_tool.GND_VERSION} unavailable")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return fx.interop_run(tmp_path_factory.mktemp("lane-x") / "interop")


def _rows(base, source="survey"):
    pop = json.loads((base / "sources" / source / "extracted.population.json").read_bytes())
    return [{"fact_type": ft, "roles": r, "span": None} for ft, rs in pop["facts"].items() for r in rs]


@pytest.fixture
def run(tmp_path, base):
    out = tmp_path / "run"
    shutil.copytree(base, out)
    shutil.rmtree(out / "artifacts")
    ex.bundle(out)
    return out


def _write(run, base, extra=()):
    document = (base / "sources/survey/document.md").read_text(encoding="utf-8")
    return ex.write_source(run, "survey", document, [*_rows(base), *extra], executing_model="mock")


def test_t_x1_extract_verify_and_anchors_pass_on_host_written_rows(run, base):
    unknown = {"fact_type": "paper_published_on", "roles": {"published": "1999-01-01", "work": "example:survey-2026"},
               "span": None}
    written = _write(run, base, [unknown])
    assert (written["rows"], written["unanchored"]) == (3, 1)  # a value the document lacks is left out, counted
    assert (run / "artifacts/extraction.orm.artifacts.json").is_file()
    assert ex.verify(run, "survey").meaning == "pass"
    assert ex.check_anchors(run, "survey").meaning == "pass"


def test_t_x2_an_edited_anchor_span_is_reported_with_its_own_meaning(run, base):
    _write(run, base)
    path = run / "sources/survey" / ex.ANCHORS
    doc = json.loads(path.read_bytes())
    doc["anchors"][0]["span"]["start"] += 1
    path.write_bytes(ex._dump(doc))
    result = ex.check_anchors(run, "survey")
    assert result.exit_code in (1, 3) and result.meaning in ("alethic-failure", "input-refused")


def test_t_x3_rebind_writes_the_record_gnd_writes_and_binds_for_assurance(run, base):
    _write(run, base)
    record = ex.rebind(run, "survey")
    assert record["schema"] == "literature-rebind/1" and record["source"] == "survey"
    # the host's rebound bytes are GND's own for the same rows
    assert (run / "sources/survey/population.json").read_bytes() == (base / "sources/survey/population.json").read_bytes()
    assert binding_refusal(run) is None


def test_t_x4_roots_with_different_include_pins_are_refused(run, base):
    _write(run, base)
    # the same vocabulary, re-serialised: identical content, another file digest, so another pin
    raw = json.dumps(json.loads((run / "vocabulary/literature.orm.json").read_bytes()), indent=2).encode()
    (run / "vocabulary/literature-b.orm.json").write_bytes(raw)
    checking = json.loads((run / ex.CHECKING).read_bytes())
    for inc in checking["includes"]:
        if inc["id"] == "literature":
            inc.update(path="vocabulary/literature-b.orm.json", sha256=ex._sha(raw))
    (run / ex.CHECKING).write_bytes(ex._dump(checking))
    with pytest.raises(ex.ExtractionError, match="rebind:refused:root vocabulary pins differ"):
        ex.rebind(run, "survey")


def test_rows_from_reading_keep_only_facts_that_name_their_roles():
    facts = [{"fact_type": "a", "roles": {"x": "1"}, "span": {"start": 0, "end": 1}}, {"fact_type": "b", "roles": {}},
             {"fact_type": "c", "predicate": "p"}]
    rows, skipped = ex.rows_from_reading(facts)
    assert rows == [{"fact_type": "a", "roles": {"x": "1"}, "span": {"start": 0, "end": 1}}] and skipped == 2


def test_review_f2_a_value_found_only_outside_its_evidence_span_is_unanchored(run, base):
    rows = _rows(base)
    narrowed = [{**rows[0], "span": {"start": 0, "end": 1}}, *rows[1:]]
    document = (base / "sources/survey/document.md").read_text(encoding="utf-8")
    written = ex.write_source(run, "survey", document, narrowed, executing_model="mock")
    assert written["unanchored"] == 1


def test_review_r2_f3_a_reading_span_is_moved_into_gnd_nfc_offsets(run):
    document = "e\u0301\nGridNet uses multigrid.\n"
    (run / "nfc.md").write_text(document, encoding="utf-8")
    start = document.index("GridNet")
    row = {"fact_type": "paper_names_parent_method", "roles": {"method": "GridNet"},
           "span": {"start": start, "end": start + len("GridNet uses multigrid.")}}
    out = ex._helper(run, "anchors", {"model": str(run / ex.EXTRACTION), "document": str(run / "nfc.md"),
                                      "rows": [row]})
    assert out["unanchored"] == [] and out["anchors"][0]["span"] == {"start": start - 1, "end": start - 1 + 7}
