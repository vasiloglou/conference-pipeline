"""v6.1 Phase 2: the work-family roster (spec 6.1, tests T1-T6b, T22)."""

from __future__ import annotations

import json

import pytest

from conference_pipeline.ontology import work_families as wf
from tests._ontology_fixtures import POSTER_BOARD, TITLE_A, aid, build_workspace


@pytest.fixture
def workspace(tmp_path):
    return build_workspace(tmp_path)


@pytest.fixture
def roster(workspace):
    return wf.build_roster(workspace, run_id="t")


def _family_of(roster, artifact):
    found = [f for f in roster["families"] if artifact in f["artifacts"]]
    assert len(found) == 1, (artifact, found)
    return found[0]


def _unjoined(roster):
    return {row["artifact_id"]: row["reason"] for row in roster["unjoined"]}


# T1
def test_main_paper_joins_its_forum(roster):
    family = _family_of(roster, aid("paper_text:a01"))
    assert family["family_key"] == "openreview:FORUMAAAA1"
    assert family["kind"] == "main-paper" and family["track"] == "Conf regular"
    assert family["topics"] == ["t01_alpha"]


def test_main_paper_without_an_id_field_joins_by_its_attachment_url(roster):
    family = _family_of(roster, aid("paper_text:b01"))
    assert family["family_key"] == "openreview:FORUMBBBB2"
    assert family["track"] == "Conf spotlight" and family["authors"] == ["Cy Gamma"]


# T2
def test_workshop_paper_is_its_own_family(roster):
    family = _family_of(roster, aid("workshop_paper_text:w01"))
    assert family["family_key"] == "openreview:WSPAPER001"
    assert family["kind"] == "workshop-paper" and family["track"] == "workshop:WS/One"
    assert family["authors"] == ["Wen Ye", "Ola Ade"]


# T3 (O7)
def test_poster_slides_and_transcript_join_the_paper_and_carry_its_authors(roster):
    family = _family_of(roster, aid("paper_text:p01"))
    assert family["family_key"] == "openreview:FORUMAAAA1"
    assert set(family["artifacts"]) == {aid("paper_text:a01"), aid("paper_text:p01"), aid("speech_transcript:s01")}
    assert family["authors"] == ["Ann Alpha", "Bo Beta"] and family["title"] == TITLE_A
    assert family["artifact_kinds"][aid("paper_text:p01")] == "paper_text:slides:poster"


def test_a_poster_with_no_paper_is_a_companion_family(roster):
    family = _family_of(roster, aid("speech_transcript:s02"))
    assert family["kind"] == "companion" and family["family_key"] == "site:poster-102"


def test_a_paper_with_no_stable_id_is_a_site_paper(roster):
    family = _family_of(roster, aid("paper_text:c90"))
    assert family["kind"] == "site-paper" and family["track"] == "site:calendar"


# T3b (D3)
def test_one_work_in_two_tracks_is_two_families_in_one_split_group(roster):
    main = _family_of(roster, aid("paper_text:c01"))
    workshop = _family_of(roster, aid("workshop_paper_text:w02"))
    assert main["family_key"] != workshop["family_key"]
    assert main["split_group"] == workshop["split_group"]
    assert roster["title_collisions"] and roster["title_collisions"][0]["paper_keys"] == sorted(
        [main["family_key"], workshop["family_key"]])


# T4
def test_malformed_keyless_and_non_indexable_records_are_unjoined_with_reasons(roster):
    unjoined = _unjoined(roster)
    assert unjoined[aid("media_asset:m01")] == "not-indexable"
    assert unjoined["record[19]"] == "malformed" and unjoined["record[20]"] == "malformed"
    assert unjoined[aid("paper_text:k01")] == "no-key"
    assert unjoined[aid("speech_transcript:k02")] == "no-key"


# T5
def test_the_poster_board_record_is_not_a_work(roster):
    assert _unjoined(roster)[POSTER_BOARD] == "not-a-work"


def test_without_the_exclusion_list_the_poster_board_record_is_keyless(tmp_path):
    roster = wf.build_roster(build_workspace(tmp_path, exclusions=False), run_id="t")
    assert _unjoined(roster)[POSTER_BOARD] == "no-key"


# T6
def test_roster_problems_is_empty_for_a_built_roster(workspace, roster):
    assert wf.roster_problems(roster, wf.inventory_artifact_ids(workspace)) == []


def test_roster_problems_names_a_missing_and_a_doubled_artifact(workspace, roster):
    broken = json.loads(json.dumps(roster))
    dropped = broken["families"][0]["artifacts"].pop()
    broken["families"][1]["artifacts"].append(broken["families"][2]["artifacts"][0])
    problems = wf.roster_problems(broken, wf.inventory_artifact_ids(workspace))
    assert f"artifact {dropped} is in no family and not unjoined" in problems
    assert any(p.startswith(f"artifact {broken['families'][2]['artifacts'][0]} appears 2 times") for p in problems)


def test_roster_problems_names_a_family_without_a_group_and_an_unjoined_row_without_a_reason(workspace, roster):
    broken = json.loads(json.dumps(roster))
    broken["families"][0]["split_group"] = ""
    broken["unjoined"][0]["reason"] = ""
    problems = wf.roster_problems(broken, wf.inventory_artifact_ids(workspace))
    assert any("has no split group" in p for p in problems)
    assert any("has no reason" in p for p in problems)


# T6b (D4)
def test_an_oral_session_links_its_papers_and_shares_their_group(roster):
    session = _family_of(roster, aid("speech_transcript:s03"))
    assert session["kind"] == "event" and session["track"] == "event:oral"
    assert session["linked_families"] == ["openreview:FORUMDDDD4", "openreview:FORUMEEEE5"]
    assert session["unlinked_talks"] == 2  # the scheduled title and the opening remarks
    groups = {session["split_group"]} | {f["split_group"] for f in roster["families"]
                                         if f["family_key"] in session["linked_families"]}
    assert len(groups) == 1


def test_oral_slides_join_their_paper_by_item_id_and_title(roster):
    assert _family_of(roster, aid("paper_text:o01"))["family_key"] == "openreview:FORUMDDDD4"


def test_single_events_group_their_materials_by_item(roster):
    tutorial = _family_of(roster, aid("paper_text:t01"))
    assert tutorial["family_key"] == "event:virtual-5"
    assert set(tutorial["artifacts"]) == {aid("paper_text:t01"), aid("speech_transcript:s05")}
    assert _family_of(roster, aid("speech_transcript:s04"))["track"] == "event:invited-talk"


def test_a_missing_talk_map_is_counted_not_raised(roster):
    recording = _family_of(roster, aid("speech_transcript:s06"))
    assert recording["linked_families"] == [] and recording["unlinked_talks"] == 1


# Totality over whole input files
@pytest.mark.parametrize("relative", ["storage/corpus/corpus_inventory.json",
                                      "storage/download-corpus/openreview_accepted_papers.json"])
def test_a_missing_or_unreadable_input_file_is_refused_by_name(workspace, relative):
    (workspace / relative).write_text("{not json", encoding="utf-8")
    with pytest.raises(wf.OntologyInputError, match="unreadable"):
        wf.build_roster(workspace, run_id="t")
    (workspace / relative).unlink()
    with pytest.raises(wf.OntologyInputError, match="missing"):
        wf.build_roster(workspace, run_id="t")


def test_two_full_read_runs_are_refused_unless_one_is_named(workspace):
    second = workspace / "storage/extractions/run2/llm-full-read/by-unit"
    second.mkdir(parents=True)
    with pytest.raises(wf.OntologyInputError, match="2 full-read runs"):
        wf.build_roster(workspace, run_id="t")
    roster = wf.build_roster(workspace, run_id="t", full_read_dir=second)
    assert _family_of(roster, aid("paper_text:a01"))["topics"] == []


def test_the_roster_is_a_pure_function_of_its_inputs(workspace):
    first = wf.canonical_bytes(wf.build_roster(workspace, run_id="t"))
    assert wf.canonical_bytes(wf.build_roster(workspace, run_id="t")) == first


# T22 (R2): another workspace with other track, topic and workshop names
def test_a_second_workspace_with_other_names_gives_the_same_structure(tmp_path, roster):
    other = wf.build_roster(build_workspace(tmp_path / "b", prefix="Zed "), run_id="t")
    shape = [(f["family_key"], f["kind"], sorted(f["artifacts"])) for f in roster["families"]]
    assert [(f["family_key"], f["kind"], sorted(f["artifacts"])) for f in other["families"]] == shape
    family = _family_of(other, aid("workshop_paper_text:w01"))
    assert family["track"] == "workshop:Zed WS/One"
    assert _family_of(other, aid("paper_text:a01"))["topics"] == ["Zed t01_alpha"]
