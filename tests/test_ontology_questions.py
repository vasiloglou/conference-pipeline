"""v6.1 Phase 2: competency questions as one machine-readable file (spec 6.6, T19-T20)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conference_pipeline.ontology import questions as q

DRAFTS = Path(__file__).resolve().parents[1] / ".planning/milestones/v6.1-ICML-2026-Ontology/notes/competency-questions"


def _draft(tmp_path, body, name="area-01.md"):
    (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


_TWO = """# Area 01
### Q-01-1. First question text?
- Kind: lookup
- Forces: X
- Tier: B
- Priority: core
- Evidence: "x" 3

### Q-01-2. Second question text?
- Kind: contradiction
- Tier: C
- Priority: extended
"""


# T19
def test_every_drafted_question_appears_once_with_its_priority():
    document = q.build_questions(DRAFTS)
    headings = [m for path in sorted(DRAFTS.glob("*.md")) for m in
                re.findall(r"^### (Q-[0-9A-Z]+-\d+)\.", path.read_text(encoding="utf-8"), flags=re.M)]
    ids = [x["id"] for x in document["questions"]]
    assert sorted(ids) == sorted(headings) and len(set(ids)) == len(ids)
    assert all(x["priority"] in {"core", "extended"} for x in document["questions"])


@pytest.mark.repo_data(".planning/milestones/v6.1-ICML-2026-Ontology/notes/competency-questions/core-set.md")
def test_the_approved_core_set_is_present_as_core():
    document = q.build_questions(DRAFTS, DRAFTS / "edits.json")
    core = q.core_ids_from_set(DRAFTS / "core-set.md")
    assert core and q.question_problems(document, core) == []
    assert document["counts"]["core"] == len(core)


def test_question_problems_names_a_missing_or_demoted_core_question(tmp_path):
    document = q.build_questions(_draft(tmp_path, _TWO))
    assert q.question_problems(document, ["Q-01-1", "Q-01-2", "Q-01-9"]) == [
        "Q-01-2 is listed as core but has priority 'extended'", "core question Q-01-9 is missing"]


def test_parsing_keeps_fields_and_refuses_a_duplicate_id(tmp_path):
    document = q.build_questions(_draft(tmp_path, _TWO))
    first = document["questions"][0]
    assert (first["id"], first["area"], first["kind"], first["tier"], first["evidence"]) == (
        "Q-01-1", "01", "lookup", "B", '"x" 3')
    _draft(tmp_path, _TWO.replace("Area 01", "Again"), name="area-02.md")
    with pytest.raises(q.QuestionsError, match="Q-01-1 appears in"):
        q.build_questions(tmp_path)


# T20
def _edits(tmp_path, edits):
    path = tmp_path / "edits.json"
    path.write_text(json.dumps({"edits": edits}), encoding="utf-8")
    return path


def test_an_approved_merge_keeps_the_id_and_names_its_target(tmp_path):
    drafts = _draft(tmp_path, _TWO)
    document = q.build_questions(drafts, _edits(tmp_path, [
        {"kind": "merge", "question": "Q-01-2", "into": "Q-01-1", "status": "approved"}]))
    merged = {x["id"]: x["merged_into"] for x in document["questions"]}
    assert merged == {"Q-01-1": None, "Q-01-2": "Q-01-1"}
    assert document["counts"]["merged"] == 1


def test_a_proposed_or_rejected_edit_changes_nothing(tmp_path):
    drafts = _draft(tmp_path, _TWO)
    for status in ("proposed", "rejected"):
        document = q.build_questions(drafts, _edits(tmp_path, [
            {"kind": "merge", "question": "Q-01-2", "into": "Q-01-1", "status": status},
            {"kind": "reword", "question": "Q-01-1", "text": "New?", "status": status}]))
        assert all(x["merged_into"] is None and x["original_text"] is None for x in document["questions"])
    assert document["counts"]["edits_pending"] == 0


def test_an_approved_rewording_keeps_the_original_text(tmp_path):
    document = q.build_questions(_draft(tmp_path, _TWO), _edits(tmp_path, [
        {"kind": "reword", "question": "Q-01-1", "text": "  Reworded question?  ", "status": "approved"}]))
    first = document["questions"][0]
    assert (first["text"], first["original_text"]) == ("Reworded question?", "First question text?")


@pytest.mark.parametrize("edit,message", [
    ({"kind": "merge", "question": "Q-01-2", "into": "Q-09-9", "status": "proposed"}, "unknown question"),
    ({"kind": "merge", "question": "Q-01-2", "into": "Q-01-2", "status": "approved"}, "itself"),
    ({"kind": "reword", "question": "Q-01-1", "status": "approved"}, "without text"),
    ({"kind": "split", "question": "Q-01-1", "status": "approved"}, "unknown kind"),
    ({"kind": "merge", "question": "Q-01-1", "into": "Q-01-2", "status": "maybe"}, "status"),
])
def test_malformed_edits_are_refused(tmp_path, edit, message):
    with pytest.raises(q.QuestionsError, match=message):
        q.build_questions(_draft(tmp_path, _TWO), _edits(tmp_path, [edit]))


def test_a_chained_merge_is_refused(tmp_path):
    body = _TWO + "\n### Q-01-3. Third?\n- Priority: extended\n"
    with pytest.raises(q.QuestionsError, match="already merged"):
        q.build_questions(_draft(tmp_path, body), _edits(tmp_path, [
            {"kind": "merge", "question": "Q-01-2", "into": "Q-01-1", "status": "approved"},
            {"kind": "merge", "question": "Q-01-3", "into": "Q-01-2", "status": "approved"}]))


def test_the_milestone_edits_file_names_only_real_questions():
    document = q.build_questions(DRAFTS, DRAFTS / "edits.json")
    assert document["counts"]["edits_pending"] + sum(
        e["status"] != "proposed" for e in document["edits"]) == len(document["edits"])
