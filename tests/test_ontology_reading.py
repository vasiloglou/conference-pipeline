"""v6.1 Phase 4 lane R: the reading host (T-R1..T-R9)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conference_pipeline.ontology import reading as rd
from conference_pipeline.ontology.producer import Response

FIX = Path(__file__).parent / "fixtures" / "ontology" / "phase4"
PAPER = (FIX / "paper-a.txt").read_text(encoding="utf-8")
STUB = (FIX / "stub.txt").read_text(encoding="utf-8")
FACT_TYPES = ["method_uses_technique", "result_on_benchmark", "limitation"]
QUESTIONS = [
    {"id": "Q-1", "text": "Which solvers use multigrid?", "terms": ["multigrid"], "required": True},
    {"id": "Q-2", "text": "What error on Darcy?", "terms": ["darcy"], "required": True},
    {"id": "Q-3", "text": "Where do solvers fail?", "terms": ["turbulent"], "required": False},
    {"id": "Q-4", "text": "Sandbox escape rates?", "terms": ["sandbox escape"], "required": False},
]
# Section id -> facts the fake reader reports when the section is in its prompt.
FACTS = {
    "s01": [("method_uses_technique", "uses", "asserted", "We propose GridNet, a multigrid solver for elliptic equations.")],
    "s06": [("result_on_benchmark", "reaches", "asserted", "GridNet reaches an error of 0.012 on the Darcy benchmark")],
    "s07": [("limitation", "does not improve", "negated", "GridNet does not improve on turbulent flows")],
}


class Fake:
    def __init__(self, refuse=(), empty=(), keys=None, reject=()):
        self.refuse, self.empty, self.keys, self.reject, self.calls = set(refuse), set(empty), keys or {}, reject, []

    def __call__(self, role, kind, work, prompt):
        self.calls.append((role, kind, work))
        if kind in self.refuse:
            return Response("", stop_reason="refusal")
        if kind.endswith("review"):
            ids = re.findall(r'"id": "([^"]+)"', prompt.split("Excerpts:")[0])
            return Response(json.dumps({"verdicts": [
                {"id": i, "verdict": "unsupported" if i in self.reject else "supported"} for i in ids]}))
        sections = re.findall(r"^\[(\w+)\] ", prompt, re.M)
        facts = [] if kind in self.empty else [
            {"fact_type": ft, "section": s, "predicate": p, "participants": [{"role": "subject", "name": "GridNet"}],
             "polarity": pol, "quote": q} for s in sections for ft, p, pol, q in FACTS.get(s, [])]
        return Response(json.dumps({"facts": facts, "proposal_keys": self.keys.get(kind, [])}))


def _run(tmp_path, producer, documents=None, **kw):
    return rd.run(tmp_path / "run", documents or {"paper-a": PAPER}, questions=QUESTIONS, fact_types=FACT_TYPES,
                  producer=producer, **kw)


def _rows(report, document="paper-a"):
    return {r["question"]: r["state"] for r in report["coverage"] if r["document"] == document}


def test_t_r1_inventory_finds_sections_tables_and_equations():
    inv = rd.inventory(PAPER)
    by = {s["id"]: s for s in inv["sections"]}
    assert inv["structure"] == "headings"
    assert by["s01"]["heading"] == "Abstract" and by["s01"]["tier"] == "A"
    assert by["s05"]["heading"].startswith("3.1.") and by["s05"]["tier"] == "B"  # inherits "3. Experiments"
    assert by["s03"]["tier"] == "C" and by["s08"]["tier"] == "-"  # method; references never read
    table = PAPER[by["t01"]["start"]:by["t01"]["end"]]
    assert table.startswith("Table 1.") and "Baseline 0.030" in table and "4. Conclusion" not in table
    assert PAPER[by["e01"]["start"]:by["e01"]["end"]].endswith("(1)")
    for kind in ("section", "table", "equation"):
        spans = sorted((s["start"], s["end"]) for s in inv["sections"] if s["kind"] == kind)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_t_r2_a_text_without_headings_gets_windows():
    inv = rd.inventory(STUB)
    assert inv["structure"] == "windows" and [s["kind"] for s in inv["sections"]] == ["window"]


def test_t_r3_mock_run_completes_required_coverage(tmp_path):
    fake = Fake()
    report = _run(tmp_path, fake)
    (doc,) = report["documents"]
    assert doc["stop"] == "required-coverage-completed"
    assert _rows(report) == {"Q-1": "supported", "Q-2": "supported", "Q-3": "explicitly-denied", "Q-4": "not-applicable"}
    files = sorted(p.name for p in (tmp_path / "run/documents/paper-a").iterdir())
    assert files == sorted([f"stage-{i}.json" for i in range(6)] + [f"payload-{i}.json" for i in range(6)])
    chain = [json.loads((tmp_path / f"run/documents/paper-a/stage-{i}.json").read_text()) for i in range(6)]
    assert [c["stage"] for c in chain] == list(rd.STAGES) and chain[0]["predecessor_sha256"] is None
    assert all(c["predecessor_sha256"] == rd.sha256((tmp_path / f"run/documents/paper-a/stage-{i}.json").read_bytes())
               for i, c in enumerate(chain[1:]))
    assert doc["producer_calls"] == {"broad-extraction": 1, "targeted-extraction": 3, "review": 1, "recheck": 0}
    cov = json.loads((tmp_path / "run/coverage.json").read_text())
    assert cov["format"] == "literature-reading-coverage/1" and {r["state"] for r in cov["rows"]} <= set(rd.COVERAGE_STATES)


def test_t_r4_a_targeted_pass_that_adds_nothing_stops_the_document(tmp_path):
    report = _run(tmp_path, Fake(empty={"targeted"}))
    assert report["documents"][0]["stop"] == "targeted-pass-added-zero"
    assert _rows(report)["Q-2"] == "empty-extraction"


def test_t_r5_replay_re_derives_with_zero_calls(tmp_path):
    first = _run(tmp_path, Fake(keys={"targeted": ["abstract"]}))
    again = rd.replay(tmp_path / "run", {"paper-a": PAPER}, questions=QUESTIONS, fact_types=FACT_TYPES)
    assert set(again["producer_calls"].values()) == {0}
    assert again["evidence_sha256"] == first["evidence_sha256"] and again["coverage"] == first["coverage"]
    assert first["producer_calls"]["recheck"] >= 1  # the proposal key sent the abstract back for a recheck


@pytest.mark.parametrize("damage, reason", [
    ("edit-payload", "stale retained reading output"),
    ("delete-stage", "incomplete stage roster"),
    ("edit-stage", "stale retained reading stage"),
    ("change-text", "stale reading input"),
])
def test_t_r6_damaged_evidence_or_changed_input_is_refused(tmp_path, damage, reason):
    _run(tmp_path, Fake())
    folder, text = tmp_path / "run/documents/paper-a", PAPER
    if damage == "edit-payload":
        (folder / "payload-3.json").write_text("{}")
    elif damage == "delete-stage":
        (folder / "stage-2.json").unlink()
    elif damage == "edit-stage":
        (folder / "stage-4.json").write_text("{}")
    else:
        text = PAPER.replace("0.012", "0.013")
    with pytest.raises(rd.ReadingError, match=reason):
        rd.replay(tmp_path / "run", {"paper-a": text}, questions=QUESTIONS, fact_types=FACT_TYPES)
    args = ["--documents", str(_doclist(tmp_path, text)), "--questions", str(_qfile(tmp_path)),
            "--fact-types", str(_ftfile(tmp_path)), str(tmp_path / "run")]
    assert rd.main(["replay", *args]) == 3


def test_t_r7_budget_zero_omits_every_section_and_calls_nothing(tmp_path):
    fake = Fake()
    report = _run(tmp_path, fake, budget=0)
    assert fake.calls == [] and report["documents"][0]["stop"] == "budget-exhausted"
    assert {_rows(report)[q] for q in ("Q-1", "Q-2", "Q-3")} == {"omitted"}


def test_t_r8_a_missing_text_is_unavailable_without_a_call(tmp_path):
    fake = Fake()
    report = _run(tmp_path, fake, documents={"paper-a": PAPER, "gone": None})
    assert all(c[2].startswith("paper-a#") for c in fake.calls)
    assert set(_rows(report, "gone").values()) == {"unavailable"}


def test_t_r9_refused_and_empty_are_coverage_states_never_retried(tmp_path):
    fake = Fake(refuse={"broad"})
    report = _run(tmp_path, fake)
    assert _rows(report)["Q-1"] == "producer-refused"
    assert sum(c[1] == "broad" for c in fake.calls) == 1
    report = rd.run(tmp_path / "run2", {"paper-a": PAPER}, questions=QUESTIONS, fact_types=FACT_TYPES,
                    producer=Fake(empty={"broad"}))
    assert _rows(report)["Q-1"] == "empty-extraction"


def test_review_rejection_drops_the_fact_and_sends_its_section_to_recheck(tmp_path):
    fake = Fake(reject={"b01"})
    report = _run(tmp_path, fake)
    assert any(c[1] == "recheck" for c in fake.calls) and any(c[1] == "recheck-review" for c in fake.calls)
    assert _rows(report)["Q-1"] == "supported"  # the recheck found it again and its review passed


def test_a_quote_not_in_its_section_is_dropped(tmp_path):
    raw = {"text": json.dumps({"facts": [{"fact_type": "limitation", "section": "s01", "quote": "not in this paper at all"}]}),
           "stop_reason": "end", "provider": "p", "model": "m"}
    sections = rd.inventory(PAPER)["sections"]
    outcome, facts, drops, _ = rd.read_facts(raw, PAPER, sections, FACT_TYPES, "x")
    assert (outcome, facts, drops) == ("empty", [], [{"fact": "x01", "outcome": "quote-not-in-section"}])


def test_question_terms_from_the_competency_file():
    doc = {"questions": [{"id": "Q-1", "text": "t", "evidence": '"conservation law" 29; "weak form|weak-form" 4',
                          "priority": "core", "merged_into": None},
                         {"id": "Q-2", "text": "t", "evidence": '"x y z" 1', "priority": "extended", "merged_into": "Q-1"}]}
    assert rd.question_terms(doc) == [{"id": "Q-1", "text": "t", "terms": ["conservation law", "weak form", "weak-form"],
                                       "required": True}]


def test_cli_run_with_saved_responses_and_usage_error(tmp_path):
    out = tmp_path / "cli"
    args = ["--documents", str(_doclist(tmp_path, PAPER)), "--questions", str(_qfile(tmp_path)),
            "--fact-types", str(_ftfile(tmp_path))]
    assert rd.main(["run", *args, "--out", str(out), "--responses", str(tmp_path / "none")]) == 0
    assert rd.main(["replay", *args, str(out)]) == 0
    assert rd.main(["bogus"]) == 64


def _doclist(tmp_path, text):
    (tmp_path / "paper-a.txt").write_text(text, encoding="utf-8")
    path = tmp_path / "docs.tsv"
    path.write_text(f"paper-a\t{tmp_path / 'paper-a.txt'}\n", encoding="utf-8")
    return path


def _qfile(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": [{"id": q["id"], "text": q["text"], "merged_into": None,
                                               "priority": "core" if q["required"] else "extended",
                                               "evidence": "; ".join(f'"{t}" 1' for t in q["terms"])}
                                              for q in QUESTIONS]}), encoding="utf-8")
    return path


def _ftfile(tmp_path):
    path = tmp_path / "fact-types.json"
    path.write_text(json.dumps(FACT_TYPES), encoding="utf-8")
    return path


# ---- Phase 4 review r1 regressions


@pytest.mark.parametrize("outcome", ["refuse", "empty"])
def test_review_f6_a_refused_or_empty_section_is_not_retried_by_recheck(tmp_path, outcome):
    fake = Fake(**{outcome: {"broad"}}, keys={"targeted": ["abstract"]})
    report = _run(tmp_path, fake)
    assert not any(kind == "recheck" for _, kind, _ in fake.calls)
    assert _rows(report)["Q-1"] == ("producer-refused" if outcome == "refuse" else "empty-extraction")


def test_review_f7_conflicting_verdicts_of_any_count_leave_the_fact_unjudged():
    facts = [{"id": "b01"}]
    three = {"text": json.dumps({"verdicts": [{"id": "b01", "verdict": v}
                                              for v in ("supported", "unsupported", "supported")]}),
             "stop_reason": "end", "provider": "p", "model": "m"}
    assert rd.read_verdicts(three, facts) == {"b01": "missing"}


@pytest.mark.parametrize("tiers", [(), ("B",)])
def test_review_f8_tier_a_is_read_only_when_the_run_asks_for_it(tmp_path, tiers):
    fake = Fake()
    _run(tmp_path, fake, tiers=tiers)
    assert not any(kind == "broad" for _, kind, _ in fake.calls)


def test_review_f2_role_values_must_come_from_the_quote_and_reach_the_reviewer():
    section = next(s for s in rd.inventory(PAPER)["sections"] if s["id"] == "s01")
    quote = FACTS["s01"][0][3]
    reply = {"facts": [{"fact_type": "method_uses_technique", "section": "s01", "quote": quote,
                        "roles": {"method": "GridNet", "technique": "multigrid"}},
                       {"fact_type": "method_uses_technique", "section": "s01", "quote": quote,
                        "roles": {"method": "Baseline", "technique": "multigrid"}}]}
    raw = {"text": json.dumps(reply), "stop_reason": "end", "provider": "p", "model": "m"}
    outcome, facts, drops, _ = rd.read_facts(raw, PAPER, [section], FACT_TYPES, "b")
    assert [f["id"] for f in facts] == ["b01"] and drops == [{"fact": "b02", "outcome": "role-not-in-quote"}]
    assert '"roles": {"method": "GridNet", "technique": "multigrid"}' in rd.review_prompt(PAPER, facts, {"s01": section})
