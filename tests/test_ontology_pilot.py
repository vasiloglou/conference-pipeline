"""v6.1 Phase 5: model-pilot sample, claim-shaped reading, seed mapping, scorer and harness.

Spec ``notes/phase-5-spec.md`` §5 (T-S, T-E, T-K, T-C, T-H, T-G). GND-backed cases (T-E3,
T-E6, T-G1) run real GND 0.4.0 offline and skip without it.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline.ontology import assertions, gnd_tool, gold_mapping, pilot, pilot_score
from conference_pipeline.ontology import reading as rd
from conference_pipeline.ontology.producer import OntologyProducer, Response, SavedProducer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ontology"))
import pilot_run  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "ontology" / "phase4"
PAPER = (FIX / "paper-a.txt").read_text(encoding="utf-8")
SEED = ROOT / "src/conference_pipeline/ontology/vocabulary/seed-v1.orm.yaml"
FACT_TYPES = pilot_run.fact_types()


def _gnd_missing() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


needs_gnd = pytest.mark.skipif(_gnd_missing(), reason=f"gnd {gnd_tool.GND_VERSION} unavailable")


def _span(sentence: str) -> dict[str, int]:
    start = PAPER.index(sentence)
    return {"start": start, "end": start + len(sentence)}


def _statement(sid, sentence, predicate, participants, *, support="supported", polarity="asserted",
               conditions=(), quantity=None, passage="p01"):
    return {"id": sid, "span": _span(sentence), "predicate": predicate, "support": support, "polarity": polarity,
            "attribution": "own", "conditions": list(conditions), "experiment": None, "quantity": quantity,
            "participants": [{"name": n, "role": r} for n, r in participants], "passage": passage}


S01 = "We propose GridNet, a multigrid solver for elliptic equations."
S02 = "GridNet reaches an error of 0.012 on the Darcy benchmark"
S03 = "GridNet does not improve on turbulent flows"
S05 = "GridNet reduces error by 40% on the Poisson benchmark."


def _gold(work="openreview:A"):
    return {"work": work, "schema": "ontology-gold/1", "split": "dev",
            "passages": [{"id": "p01", "span": {"start": 54, "end": 182}},
                         {"id": "p02", "span": {"start": 540, "end": 722}},
                         {"id": "p03", "span": {"start": 722, "end": 801}}],
            "statements": [
                _statement("s01", S01, "propose", [("GridNet", "method")]),
                _statement("s02", S02, "reaches", [("GridNet", "method"), ("Darcy benchmark", "benchmark")],
                           passage="p02"),
                _statement("s03", S03, "does not improve", [("GridNet", "method")], polarity="negated",
                           conditions=["on turbulent flows"], passage="p03"),
                _statement("s04", S02, "reaches", [("GridNet", "method")], support="contradicted", passage="p02"),
                _statement("s05", S05, "reduces error", [("GridNet", "method"), ("Poisson benchmark", "benchmark")],
                           quantity={"coefficient": 40, "scale": 0, "unit": "%"})]}


def _pair(tmp_path, works=("openreview:A",), domains=("01",), sealed=True):
    pair = tmp_path / "gold" / "dev"
    (pair / "documents").mkdir(parents=True)
    (pair / "gold").mkdir()
    import hashlib
    docs = []
    for n, (work, domain) in enumerate(zip(works, domains, strict=True)):
        ident = work.replace(":", "-")
        (pair / "documents" / f"{ident}.md").write_text(PAPER, encoding="utf-8")
        (pair / "gold" / f"{ident}.gold.json").write_text(json.dumps(_gold(work)), encoding="utf-8")
        docs.append({"id": ident, "domain": domain, "path": f"documents/{ident}.md", "work": {"identifier": work},
                     "sha256": hashlib.sha256(PAPER.encode()).hexdigest(), "n": n})
    (pair / "corpus.json").write_text(json.dumps({"schema": "x", "split": "dev", "documents": docs, "passages": []}))
    (pair / "links.json").write_text(json.dumps({"name_pairs": [
        {"id": "n1", "kind": "synonym", "left": "GridNet", "right": "Grid-Net", "passages": ["openreview:A#p01"]}]}))
    if sealed:
        (tmp_path / "gold" / "sealed_test").mkdir()
        (tmp_path / "gold" / "sealed_test" / "corpus.json").write_text("{}")
    return pair


def _claim(sentence, predicate, participants, *, section, polarity="asserted", conditions=()):
    return {"section": section, "quote": sentence, "predicate": predicate, "polarity": polarity, "attribution": "own",
            "participants": [{"name": n, "role": r} for n, r in participants], "conditions": list(conditions),
            "experiment": None}


CLAIMS = {"s01": [_claim(S01, "propose", [("GridNet", "method")], section="s01")],
          "s06": [_claim(S02, "reaches", [("GridNet", "method"), ("Darcy benchmark", "benchmark")], section="s06")],
          "s07": [_claim(S03, "does not improve", [("GridNet", "method")], section="s07", polarity="negated",
                         conditions=["on turbulent flows"])]}


class Reader:
    """Answers extraction calls with CLAIMS for the sections in the prompt, review calls all-supported."""

    def __init__(self, claims=None, flip=()):
        self.claims, self.flip, self.log = claims or CLAIMS, set(flip), []

    def __call__(self, role, kind, work, prompt):
        self.log.append((role, kind, work))
        if role == "reviewer":
            ids = re.findall(r'"id": "([^"]+)"', prompt.split("Excerpts:")[0])
            return Response(json.dumps({"verdicts": [
                {"id": i, "verdict": "unsupported" if i in self.flip else "supported"} for i in ids]}))
        sections = re.findall(r"^\[(\w+)\] ", prompt, re.M)
        return Response(json.dumps({"assertions": [c for s in sections for c in self.claims.get(s, [])],
                                    "proposal_keys": []}))


# --------------------------------------------------------------------------- lane S


def test_t_s1_sample_is_one_per_area_then_salted_and_stable(tmp_path):
    works = [f"openreview:W{i}" for i in range(6)]
    pair = _pair(tmp_path, works, ("01", "01", "02", "02", "03", "03"), sealed=False)
    corpus = json.loads((pair / "corpus.json").read_bytes())
    a, b = pilot.select_sample(corpus, salt="s", n=4), pilot.select_sample(corpus, salt="s", n=4)
    assert a == b and a["sha256"] == pilot.digest(a) and len(a["works"]) == 4
    assert {w["domain"] for w in a["works"]} == {"01", "02", "03"}
    with pytest.raises(pilot.PilotError):
        pilot.select_sample({**corpus, "split": "sealed_test"}, salt="s", n=4)
    with pytest.raises(pilot.PilotError):
        pilot.check_sample({**a, "salt": "other"})


def test_t_s2_no_sealed_byte_is_read_while_sampling_running_and_scoring(tmp_path):
    pair = _pair(tmp_path)
    sealed = tmp_path / "gold" / "sealed_test"
    os.chmod(sealed / "corpus.json", 0)
    os.chmod(sealed, 0)
    try:
        sample = pilot.select_sample(json.loads((pair / "corpus.json").read_bytes()), salt="s", n=1)
        texts, golds, _ = pilot.load_dev(pair, sample)
        summary = pilot.run_arm("broad", sample, tmp_path / "run", producer=Reader(),
                                inputs={"texts": texts, "fact_types": FACT_TYPES})
        assert summary["status"] == "complete"
    finally:
        os.chmod(sealed, stat.S_IRWXU)
        os.chmod(sealed / "corpus.json", stat.S_IRUSR | stat.S_IWUSR)


# --------------------------------------------------------------------------- lane E


def _sections(ids):
    return [s for s in rd.inventory(PAPER)["sections"] if s["id"] in ids]


def test_t_e1_a_claim_shaped_reply_is_parsed_with_its_span_located():
    raw = {"text": json.dumps({"assertions": CLAIMS["s07"]}), "stop_reason": "end", "provider": "x", "model": "y"}
    outcome, facts, drops, _ = rd.read_facts(raw, PAPER, _sections({"s07"}), FACT_TYPES, "b", "assertions")
    assert outcome == "accepted" and drops == []
    (fact,) = facts
    assert fact["span"] == _span(S03) and fact["polarity"] == "negated"
    assert fact["conditions"] == ["on turbulent flows"] and fact["participants"] == [{"name": "GridNet", "role": "method"}]


def test_t_e2_names_outside_the_quote_and_values_outside_the_lists_are_dropped():
    bad_name = _claim(S03, "does not improve", [("FluxNet", "method")], section="s07")
    bad_value = {**_claim(S03, "does not improve", [("GridNet", "method")], section="s07"), "polarity": "maybe"}
    raw = {"text": json.dumps({"assertions": [bad_name, bad_value]}), "stop_reason": "end", "provider": "x", "model": "y"}
    outcome, facts, drops, _ = rd.read_facts(raw, PAPER, _sections({"s07"}), FACT_TYPES, "b", "assertions")
    assert outcome == "empty" and facts == []
    assert [d["outcome"] for d in drops] == ["role-not-in-quote", "invalid-value"]


def test_t_e4_old_runs_replay_and_a_shape_mismatch_is_refused(tmp_path):
    from tests.test_ontology_reading import FACT_TYPES as OLD_TYPES
    from tests.test_ontology_reading import QUESTIONS, Fake

    rd.run(tmp_path / "old", {"paper-a": PAPER}, questions=QUESTIONS, fact_types=OLD_TYPES, producer=Fake())
    manifest = json.loads((tmp_path / "old/run.json").read_bytes())
    assert "reply_shape" not in manifest["inputs"]  # a Phase 4 run's pins are unchanged
    assert rd.replay(tmp_path / "old", {"paper-a": PAPER}, questions=QUESTIONS, fact_types=OLD_TYPES)
    with pytest.raises(rd.ReadingError, match="stale reading input"):
        rd.replay(tmp_path / "old", {"paper-a": PAPER}, questions=QUESTIONS, fact_types=OLD_TYPES,
                  reply_shape="assertions")
    rd.run(tmp_path / "new", {"paper-a": PAPER}, questions=QUESTIONS, fact_types=FACT_TYPES, producer=Reader(),
           shape="assertions")
    assert json.loads((tmp_path / "new/run.json").read_bytes())["inputs"]["reply_shape"] == "assertions"
    assert rd.replay(tmp_path / "new", {"paper-a": PAPER}, questions=QUESTIONS, fact_types=FACT_TYPES)["documents"]


def test_t_e5_a_work_with_nine_opening_sections_is_read_in_two_broad_calls():
    text = "\n\n".join(f"{n}. Summary part {n}\nThis part states finding {n} about GridNet." for n in range(1, 10))
    assert [len(g) for g in pilot.units("broad", text)] == [6, 3]
    reader = Reader(claims={})
    rd.read_document("w", text, questions=[], fact_types=FACT_TYPES, producer=reader, tiers=("A",), shape="assertions")
    assert sum(1 for _, kind, _ in reader.log if kind == "broad") == 2


@needs_gnd
def test_t_e3_t_e6_claims_become_seed_rows_anchored_on_text_roles_and_gnd_accepts_them(tmp_path):
    raw = {"text": json.dumps({"assertions": [c for cs in CLAIMS.values() for c in cs]}), "stop_reason": "end",
           "provider": "x", "model": "y"}
    _, claims, _, _ = rd.read_facts(raw, PAPER, _sections({"s01", "s06", "s07"}), FACT_TYPES, "b", "assertions")
    rows = assertions.reading_rows("openreview:A", claims)
    assert {r["fact_type"] for r in rows} >= {"work_has_version", "assertion_in_version", "assertion_cites_passage",
                                              "assertion_has_predicate", "assertion_has_participant_in_role",
                                              "assertion_under_condition", "assertion_has_polarity"}
    run = pilot.seed_run(tmp_path / "verify", SEED)
    written = pilot.verify_claims(run, "openreview:A", PAPER, claims, executing_model="mock")
    assert written["unanchored"] == 0 and written["rows"] == len(rows)
    side = json.loads((run / "sources/openreview-A" / "extracted.population.json.anchors.json").read_bytes())
    assert {(a["fact_key"]["fact_type"], a["role"]) for a in side["anchors"]} == {
        (ft, role) for ft, roles in assertions.TEXTUAL_ROLES.items() for role in roles}
    from conference_pipeline.ontology import extraction as ex
    anchored = ex.check_anchors(run, "openreview-A")
    # T-E6: GND binds every anchor it is given and does not demand one per role. The exit is GND's
    # own (empty fact types make some constraints vacuous); it is never an anchor refusal.
    assert anchored.meaning != "input-refused" and written["verify_meaning"] != "input-refused"


# --------------------------------------------------------------------------- lane K


def test_t_k1_the_seed_mapping_of_the_gold_has_no_problems():
    gold = _gold()
    mapping = assertions.seed_mapping([gold])
    assert gold_mapping.mapping_problems(mapping, [gold], gold_mapping.model_fact_types(SEED)) == []
    by_id = {e["statement"]: e for e in mapping["statements"]}
    assert by_id["s05"]["unexpressible"]["missing"] == ["quantity"] and "facts" not in by_id["s05"]
    assert {f["fact_type"] for f in by_id["s03"]["facts"]} >= {"assertion_under_condition", "assertion_has_polarity"}


# --------------------------------------------------------------------------- lane C


def _arm(claims_by_section, tiers=("A", "B")):
    reader = Reader(claims=claims_by_section)
    out = {}
    for arm in ("broad", "targeted"):
        if arm == "targeted" and "B" not in tiers:
            continue
        result = pilot.read_arm(arm, {"openreview:A": PAPER}, producer=reader, fact_types=FACT_TYPES)["openreview:A"]
        if "openreview:A" in out:
            for k in ("calls", "claims", "drops", "sections_read"):
                out["openreview:A"][k] += result[k]
        else:
            out["openreview:A"] = result
    return out


def test_t_c1_a_run_restating_the_gold_scores_one_everywhere():
    score = pilot_score.score_reading(_arm(CLAIMS), {"openreview:A": _gold()}, {})
    # s05 has a quantity but is a supported statement in a read section: it is in the denominator
    assert score["recall"] == {"numerator": 3, "denominator": 4}
    assert score["precision"] == {"numerator": 3, "denominator": 3}
    assert all(v["numerator"] == v["denominator"] for v in score["element_agreement"].values())


def test_t_c2_a_dropped_claim_and_a_flipped_polarity_move_only_their_measure():
    base = pilot_score.score_reading(_arm(CLAIMS), {"openreview:A": _gold()}, {})
    dropped = pilot_score.score_reading(_arm({k: v for k, v in CLAIMS.items() if k != "s06"}),
                                        {"openreview:A": _gold()}, {})
    assert dropped["recall"]["numerator"] == base["recall"]["numerator"] - 1
    flipped = {**CLAIMS, "s01": [{**CLAIMS["s01"][0], "polarity": "speculated"}]}
    flip = pilot_score.score_reading(_arm(flipped), {"openreview:A": _gold()}, {})
    assert flip["recall"] == base["recall"]
    assert flip["element_agreement"]["polarity"]["numerator"] == base["element_agreement"]["polarity"]["numerator"] - 1


def test_t_c3_claims_outside_passages_are_counted_apart_and_name_pairs_align():
    method = "GridNet uses multigrid coarsening with a conservation law penalty on every level."
    extra = {"s03": [_claim(method, "uses", [("GridNet", "method")], section="s03")]}
    arm = {"openreview:A": {"calls": [], "drops": [], "sections_read": [{"id": "s03", "start": 303, "end": 426}],
                            "claims": [{**extra["s03"][0], "id": "x1", "span": _span(method)}]}}
    assert pilot_score.score_reading(arm, {"openreview:A": _gold()}, {})["outside_gold_passages"] == 1
    renamed = {**CLAIMS["s01"][0], "participants": [{"name": "GN", "role": "method"}], "id": "y", "span": _span(S01)}
    links = json.loads(json.dumps({"name_pairs": [{"kind": "synonym", "left": "GridNet", "right": "GN",
                                                   "passages": ["openreview:A#p01"]}]}))
    statements = [s for s in _gold()["statements"] if s["id"] == "s01"]
    assert pilot_score.align([renamed], statements, pilot_score._aliases(links, "openreview:A"))
    assert not pilot_score.align([renamed], statements, {})


def test_t_c4_the_review_score_is_the_hand_computed_confusion():
    arm = {"w": {"judged": [{"gold": "supported", "verdict": "supported"}, {"gold": "supported", "verdict": "unsupported"},
                            {"gold": "contradicted", "verdict": "supported"}, {"gold": "stretched", "verdict": "unsupported"},
                            {"gold": "insufficient-evidence", "verdict": "missing"}]}}
    score = pilot_score.score_review(arm)
    assert score["true_positive_rate"] == {"numerator": 1, "denominator": 2}
    assert score["true_negative_rate"] == {"numerator": 1, "denominator": 3}
    assert score["balanced_accuracy"] == round((1 / 2 + 1 / 3) / 2, 4)
    assert score["decoys_judged_supported"] == {"numerator": 1, "denominator": 3} and score["missing_verdicts"] == 1


# --------------------------------------------------------------------------- lane H


def _live(tmp_path, cap=10.0, fail=False):
    reader = Reader()

    def complete(task, *, prompt, max_tokens, temperature, caller, **options):
        if fail:
            raise RuntimeError("endpoint down")
        r = reader("reviewer" if "judge only" in prompt else "reader", "x", "w", prompt)
        return SimpleNamespace(text=r.text, stop_reason="end", provider="anthropic", model="claude-haiku-5-5",
                               input_tokens=100, output_tokens=50, receipt=None)

    return OntologyProducer(save_root=tmp_path / "responses", budget_cap_usd=cap, complete=complete,
                            estimate=lambda model, i, o, provider=None: 0.01, gpu_price=lambda p, m: None,
                            resolve=lambda task: ("anthropic", "claude-haiku-5-5"), tasks={"reader": "ontology.read_broad"})


def _sample(tmp_path, n=2):
    pair = _pair(tmp_path, [f"openreview:{c}" for c in "AB"[:n]], ("01", "02")[:n])
    return pair, pilot.select_sample(json.loads((pair / "corpus.json").read_bytes()), salt="s", n=n)


def test_t_h1_a_saved_replay_makes_no_call_and_gives_the_same_results(tmp_path):
    pair, sample = _sample(tmp_path)
    texts, _, _ = pilot.load_dev(pair, sample)
    live = _live(tmp_path)
    first = pilot.run_arm("broad", sample, tmp_path / "run1", producer=live,
                          inputs={"texts": texts, "fact_types": FACT_TYPES})
    saved = SavedProducer(tmp_path / "responses")
    second = pilot.run_arm("broad", sample, tmp_path / "run2", producer=saved,
                           inputs={"texts": texts, "fact_types": FACT_TYPES})
    assert first["calls"] == 2 and saved.calls == 0 and second["status"] == "complete"
    one, two = (json.loads((tmp_path / d / "arm.json").read_bytes())["results"] for d in ("run1", "run2"))
    assert [c["id"] for r in one.values() for c in r["claims"]] == [c["id"] for r in two.values() for c in r["claims"]]


def test_t_h2_a_cap_below_the_next_reservation_halts_and_lists_the_works_not_reached(tmp_path):
    pair, sample = _sample(tmp_path)
    texts, _, _ = pilot.load_dev(pair, sample)
    summary = pilot.run_arm("broad", sample, tmp_path / "run", producer=_live(tmp_path, cap=0.015),
                            inputs={"texts": texts, "fact_types": FACT_TYPES})
    assert summary["status"] == "budget-limited" and summary["works_done"] == [sample["works"][0]["work"]]
    assert summary["works_not_reached"] == [sample["works"][1]["work"]] and "cap" in summary["halt"]


def _fake_modal(tmp_path):
    log = tmp_path / "modal.log"
    fake = tmp_path / "modal"
    fake.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\n")
    fake.chmod(0o755)
    return log, pilot_run._stopper(SimpleNamespace(modal_app="ai4science-qwen38-bf16-serve", modal_bin=str(fake)))


def test_t_h3_the_modal_app_is_stopped_once_with_yes_even_when_the_arm_raises(tmp_path):
    pair, sample = _sample(tmp_path)
    texts, _, _ = pilot.load_dev(pair, sample)
    log, stop = _fake_modal(tmp_path)
    with pytest.raises(KeyError):  # an input missing mid-arm: the run raises, the app is still stopped
        pilot.run_arm("broad", sample, tmp_path / "run", producer=Reader(), inputs={"fact_types": FACT_TYPES},
                      stop_app=stop)
    assert log.read_text().splitlines() == ["app stop ai4science-qwen38-bf16-serve --yes"]


def test_a_provider_error_is_the_units_outcome_and_the_run_goes_on(tmp_path):
    # live pilot 2026-11-21: one HTTP 400 ended a whole candidate's run
    pair, sample = _sample(tmp_path)
    texts, _, _ = pilot.load_dev(pair, sample)
    log, stop = _fake_modal(tmp_path)
    summary = pilot.run_arm("broad", sample, tmp_path / "run", producer=_live(tmp_path, fail=True),
                            inputs={"texts": texts, "fact_types": FACT_TYPES}, stop_app=stop)
    results = json.loads((tmp_path / "run/arm.json").read_bytes())["results"]
    assert summary["status"] == "complete" and len(summary["works_done"]) == 2
    assert {c["outcome"] for r in results.values() for c in r["calls"]} == {"error"}
    assert pilot_score.score_reading(results, {w: _gold(w) for w in results}, {})["call_outcomes"] == {"error": 2}
    assert log.read_text().splitlines() == ["app stop ai4science-qwen38-bf16-serve --yes"]


def test_t_h4_gpu_charges_carry_the_surcharge_and_api_charges_do_not():
    assert pilot_score.cost(1.0, "modal", 2, 3600)["charged_usd"] == 1.05
    assert pilot_score.cost(1.0, "anthropic", 2, 3600)["charged_usd"] == 1.0
    assert pilot_score.cost(1.0, "modal", 2, 3600)["works_per_hour"] == 2.0


def test_t_h5_score_refuses_a_run_made_from_another_sample(tmp_path, capsys):
    pair, sample = _sample(tmp_path)
    texts, _, _ = pilot.load_dev(pair, sample)
    pilot.run_arm("broad", sample, tmp_path / "run", producer=Reader(), inputs={"texts": texts, "fact_types": FACT_TYPES})
    (tmp_path / "run/summary.json").write_text(json.dumps({"arm": "broad", "status": "complete", "works_done": [],
                                                           "spent_usd": 0, "wall_seconds": 1}))
    other = {**sample, "salt": "other"}
    other["sha256"] = pilot.digest(other)
    (tmp_path / "other.json").write_text(json.dumps(other))
    code = pilot_run.main(["score", "--sample", str(tmp_path / "other.json"), "--run", str(tmp_path / "run"),
                           "--dev-pair", str(pair), "--out", str(tmp_path / "score.json")])
    assert code == pilot_run.EXIT_INPUT and not (tmp_path / "score.json").exists()


def test_review_arm_judges_every_placed_gold_statement(tmp_path):
    pair, sample = _sample(tmp_path, n=1)
    texts, golds, _ = pilot.load_dev(pair, sample)
    out = pilot.review_arm(texts, golds, producer=Reader(flip={"s04"}))
    judged = {j["id"]: j["verdict"] for j in out["openreview:A"]["judged"]}
    assert judged == {"s01": "supported", "s02": "supported", "s03": "supported", "s04": "unsupported",
                      "s05": "supported"}


# --------------------------------------------------------------------------- lane G


@needs_gnd
def test_t_g1_the_growth_arm_runs_rounds_from_a_fresh_seed_on_saved_replies(tmp_path):
    from tests.test_ontology_growth import REPLY

    texts = {"openreview:G1": PAPER, "openreview:G2": (FIX / "paper-b.txt").read_text(encoding="utf-8")}
    manifest = {"schema": pilot.SAMPLE_SCHEMA, "kind": "growth", "salt": "s", "n": 2,
                "works": [{"work": w, "sha256": "x", "chars": len(t)} for w, t in texts.items()]}
    sample = {**manifest, "sha256": pilot.digest(manifest)}
    replies = iter([json.dumps({**REPLY, "document": "openreview-G1.md"}), "not json"])

    def grower(role, kind, work, prompt):
        assert role == "grower"
        return Response(next(replies))

    summary = pilot.run_arm("growth", sample, tmp_path / "run", producer=grower,
                            inputs={"seed_raw": SEED.read_bytes(), "seed_artifact": "seed", "seed_commit": "test",
                                    "text_of": texts.__getitem__,
                                    "blocked": lambda w: False, "requested_at": "2026-11-21T00:00:00+00:00"})
    rows = json.loads((tmp_path / "run/arm.json").read_bytes())["results"]
    assert summary["status"] == "complete"
    assert rows["openreview:G2"]["status"] == "rejected" and rows["openreview:G2"]["stage"] == "reply"
    model = (tmp_path / "run/growth/model.orm.yaml").read_text(encoding="utf-8")
    assert "method_evaluated_on_benchmark" in model and "openreview-G1.md" in model
    assert "assertion_has_predicate" in model and "imported" in model  # grown from the packaged seed
    assert rows["openreview:G1"]["status"] == "applied"
    assert rows["openreview:G1"]["exits"] == {"validate": 0, "preview": 0, "apply": 0}
    assert pilot_score.score_growth(list(rows.values())) == {
        "rounds": 2, "applied": 1, "rejected_by_stage": {"reply": 1}, "added_elements": 5, "open_questions": 1}


@needs_gnd
def test_t_g2_finding_f9_gnd_refuses_the_seed_as_an_unattended_root(tmp_path):
    from conference_pipeline.ontology import gnd_tool as gt
    import yaml
    model = yaml.safe_load(pilot_run.seed_text())
    model["source"] = {"kind": "prose", "mode": "unattended", "reviewer": "none",
                       "documents": [{"sha256": "0" * 64, "label": "d.md"}]}
    (tmp_path / "m.orm.yaml").write_text(yaml.safe_dump(model, sort_keys=False))
    result = gt.run_gnd(["validate", "m.orm.yaml"], family=gt.CommandFamily.CHECK, cwd=tmp_path)
    assert result.exit_code == 3 and "unattended-u2" in result.stdout + result.stderr


def test_run_arm_refuses_a_growth_sample_for_a_reading_arm(tmp_path):
    manifest = {"schema": pilot.SAMPLE_SCHEMA, "kind": "growth", "salt": "s", "n": 0, "works": []}
    with pytest.raises(pilot.PilotError):
        pilot.run_arm("broad", {**manifest, "sha256": pilot.digest(manifest)}, tmp_path / "r", producer=Reader(),
                      inputs={})



def test_a_claim_quoted_clean_from_pdf_text_with_line_breaks_is_kept():
    # live pilot 2026-11-21: "re-\ncent", "more\neasily" in the text, quoted clean by the model
    text = "1. Introduction\nRe-\ncent studies show that audio inputs can more\neasily elicit harmful responses.\n"
    section = [s for s in rd.inventory(text)["sections"] if s["tier"] == "A"]
    claim = {"section": section[0]["id"], "quote": "audio inputs can more\neasily elicit harmful responses",
             "predicate": "can more easily elicit", "participants": [{"name": "audio inputs", "role": "subject"}],
             "conditions": [], "polarity": "asserted", "attribution": "cited"}
    raw = {"text": json.dumps({"assertions": [claim]}), "stop_reason": "end", "provider": "x", "model": "y"}
    outcome, facts, drops, _ = rd.read_facts(raw, text, section, FACT_TYPES, "b", "assertions")
    assert outcome == "accepted" and drops == [] and facts[0]["predicate"] == "can more easily elicit"
    assert rd.flat("ab-\nlation") == rd.flat("ablation") and rd.flat("toover-refusal") == rd.flat("to over-refusal")


@needs_gnd
def test_t_g3_the_packaged_seed_is_a_valid_unattended_root_keeping_every_element(tmp_path):
    import yaml

    from conference_pipeline.ontology import gnd_tool as gt
    from conference_pipeline.ontology.seed_package import unattended_seed

    raw = SEED.read_bytes()
    (tmp_path / "seed-v1.orm.yaml").write_bytes(raw)
    root = unattended_seed(raw, label="seed-v1.orm.yaml", artifact="seed", commit="test")
    (tmp_path / "model.orm.yaml").write_text(root)
    result = gt.run_gnd(["validate", "model.orm.yaml", "--document", "seed-v1.orm.yaml", "--require-documents"],
                        family=gt.CommandFamily.CHECK, cwd=tmp_path)
    assert result.exit_code == 0, result.stdout + result.stderr
    seed, packed = yaml.safe_load(raw), yaml.safe_load(root)
    ids = lambda m: ([o["id"] for o in m["object_types"]], [f["id"] for f in m["fact_types"]],
                     [c["id"] for f in m["fact_types"] for c in f.get("constraints") or []] + [c["id"] for c in m["constraints"]])
    assert ids(packed) == ids(seed)
    every = [c for f in packed["fact_types"] for c in f.get("constraints") or []] + packed["constraints"]
    assert {(c["source"], c["status"], c["modality"]) for c in every} == {("imported", "proposed", "deontic")}
    assert all(c["provenance"]["anchor"] in raw.decode() for c in every)


@pytest.mark.parametrize("predicate, quote", [
    # real predicate-only rejections from the 2026-11-21 pilot replies, accepted under P5-D10
    ("achieves decisive improvements over", "RAI achieves decisive improvements across all architectures"),
    ("restores approximate monotonicity", "restore (approximate) monotonicity"),
    ("uses as baselines", "we use the vanilla adapted defenses as the baselines"),
    ("improves mean output length by 148%", "improves the mean output length of the base model by 148%"),
    ("can more easily elicit", "audio inputs can more\neasily elicit harmful responses"),
])
def test_p5_d10_a_compressed_predicate_whose_content_words_are_in_the_quote_is_kept(predicate, quote):
    assert rd.predicate_in_quote(predicate, quote)


@pytest.mark.parametrize("predicate, quote", [
    ("outperforms", "GridNet matches FluxNet on turbulent flows"),  # a verb the sentence does not have
    ("achieves win rate", "GPT-5 securing 74.6% of rounds"),
    ("does not improve", "GridNet improves accuracy"),  # a negation is content, never a small word
])
def test_p5_d10_a_predicate_bringing_in_a_word_the_quote_lacks_is_dropped(predicate, quote):
    assert not rd.predicate_in_quote(predicate, quote)


def test_p5_d10_the_predicate_rule_drops_the_claim_in_the_reader():
    claim = _claim(S03, "outperforms", [("GridNet", "method")], section="s07")
    raw = {"text": json.dumps({"assertions": [claim]}), "stop_reason": "end", "provider": "x", "model": "y"}
    outcome, facts, drops, _ = rd.read_facts(raw, PAPER, _sections({"s07"}), FACT_TYPES, "b", "assertions")
    assert outcome == "empty" and [d["outcome"] for d in drops] == ["role-not-in-quote"]


# --------------------------------------------------------------------------- PR 50 review round 1


@needs_gnd
def test_pr50_f1_a_compressed_predicate_kept_by_p5_d10_is_not_left_unanchored(tmp_path):
    claim = _claim(S01, "proposes solver", [("GridNet", "method")], section="s01")
    assert "proposes solver" not in S01.lower() and rd.predicate_in_quote("proposes solver", S01)
    raw = {"text": json.dumps({"assertions": [claim]}), "stop_reason": "end", "provider": "x", "model": "y"}
    outcome, claims, drops, _ = rd.read_facts(raw, PAPER, _sections({"s01"}), FACT_TYPES, "b", "assertions")
    assert outcome == "accepted" and drops == []
    run = pilot.seed_run(tmp_path / "verify", SEED)
    written = pilot.verify_claims(run, "openreview:A", PAPER, claims, executing_model="mock")
    assert written["unanchored"] == 0 and written["rows"] == len(assertions.reading_rows("openreview:A", claims))


def test_pr50_f2_the_predicate_tokenizer_leaves_phase_4_heading_keys_unchanged():
    assert rd._keys("Introduction to model-based learning") == {"introduction", "model-based", "learning"}
    assert rd._words("model-based") == ["model", "based"]  # the P5-D10 predicate words still split


# --------------------------------------------------------------------------- PR 50 review round 2


@pytest.mark.parametrize("predicate, quote", [
    ("does not improve", "GridNet notes improved accuracy."),  # "notes" stemmed to "not"
    ("never improves", "Nevertheless, GridNet improves accuracy."),  # "never" a prefix of "nevertheless"
    ("has no effect", "GridNet has a notable effect."),  # "no" against "notable"
    ("works without labels", "GridNet works with labels."),
])
def test_pr50_f7_a_negation_the_quote_does_not_state_is_dropped(predicate, quote):
    assert not rd.predicate_in_quote(predicate, quote)


@pytest.mark.parametrize("predicate, quote", [
    ("does not improve", "GridNet doesn't improve accuracy."),
    ("does not improve", "GridNet doesn’t improve accuracy."),
    ("cannot recover", "GridNet cannot recover the field."),
    ("can not recover", "GridNet can't recover the field."),
    ("never improves", "GridNet never improves accuracy."),
])
def test_pr50_f7_a_negation_the_quote_states_is_kept(predicate, quote):
    assert rd.predicate_in_quote(predicate, quote)


def test_pr50_f7_an_unsupported_negated_claim_is_dropped_in_the_reader():
    quote = "GridNet notes improved accuracy."
    paper = PAPER + "\n\n" + quote
    start = paper.index(quote)
    sections = [{"id": "s99", "start": start, "end": start + len(quote)}]
    claim = _claim(quote, "does not improve", [("GridNet", "method")], section="s99", polarity="negated")
    raw = {"text": json.dumps({"assertions": [claim]}), "stop_reason": "end", "provider": "x", "model": "y"}
    outcome, facts, drops, _ = rd.read_facts(raw, paper, sections, FACT_TYPES, "b", "assertions")
    assert facts == [] and [d["outcome"] for d in drops] == ["role-not-in-quote"]


@pytest.mark.parametrize("predicate, quote", [
    # the Snowball stemmer joins forms the hand-written suffix stripper missed (2026-11-21 pilot replies)
    ("applies three simultaneous constraints", "we apply three distinct and simultaneous constraints"),
    ("is used to perform RL", "we use the refined dataset VRM-Dataset-RL (40k) to perform RL."),
    ("varies number of preferences", "varying the number of preferences and polarization q"),
    # UAX #29 word boundaries split punctuation glued by PDF extraction; colons are tailored to separate
    ("fails to win", "the best model (Claude Sonnet 4.5)fails to win a single round"),
    ("performance comparison", "Table 1.Performance comparison between EpiCoCo and baseline methods"),
    ("measured by", "Harmfulness:measured by attack success rate"),
    ("approaches the weight q", "the long-run mixture approaches the reward-sampling weight q."),
])
def test_pr50_f7_words_come_from_unicode_word_boundaries_and_the_snowball_stemmer(predicate, quote):
    assert rd.predicate_in_quote(predicate, quote)


def test_pr50_f7_contractions_come_from_the_english_contraction_table():
    assert rd._words("GridNet doesn't, can't and won't") == ["gridnet", "does", "not", "can", "not", "and", "will", "not"]
    assert rd._words("cannot") == ["can", "not"]
