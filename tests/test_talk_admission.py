"""The talk-admission corpus, cases C1-C9 (v6.0-Public-Release Phase 5.1).

Each test is one case of ``tests/fixtures/talk_admission/README.md``; its answers come
from ``EXPECTED.json``. C8 compares with ``p5_output_digests.json``, the digests P5's
unmodified code wrote for its own corpus (captured before this phase changed any code).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests import test_paper_admission as p5
from tests.talk_corpus_builders import (
    HALLUCINATION,
    add_presentation,
    add_v15,
    add_v15_non_talk,
    declare_sources,
    exclude,
    paper_base,
    presentation_file,
    read_json,
    transcript_text,
    v15_transcript_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "tests" / "fixtures" / "talk_admission"
CASES = json.loads((CORPUS / "EXPECTED.json").read_text(encoding="utf-8"))["cases"]
P5_DIGESTS = json.loads((CORPUS / "p5_output_digests.json").read_text(encoding="utf-8"))["cases"]
OUTPUTS = ("corpus_inventory.json", "paper_identity_report.json", "corpus_statistics.json")


def lanes_module():
    return importlib.import_module("conference_pipeline.indexing.talk_source_lanes")


def bind(root: Path, **kwargs: Any):
    return lanes_module().bind_talk_sources(root, **kwargs)


def admit(root: Path):
    return importlib.import_module("conference_pipeline.production.paper_admission").admit_paper_sets(root)


def receipt(root: Path, lane_id: str) -> dict[str, Any]:
    return read_json(root / "storage" / "transcription-ocr" / "evidence-lanes" / lane_id / "lane_receipt.json")


def registry_lanes(root: Path) -> list[str]:
    return [lane["lane_id"] for lane in read_json(root / "storage" / "transcription-ocr" / "evidence_lanes.json")["lanes"]]


def gaps(root: Path, lane_id: str) -> list[dict[str, str]]:
    return [{"video_id": g["video_id"], "reason": g["reason"]} for g in receipt(root, lane_id)["coverage_gaps"]]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def assert_record(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if key == "transcript_equals":
            continue
        assert actual.get(key) == value, key


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "conf"
    path.mkdir()
    return path


def test_c1_one_deep_dive_talk(root: Path) -> None:
    add_presentation(root, "1001", kind="oral", title="Scaling Laws for Sparse Experts", slideslive="5001")
    declare_sources(root, deep_dive=True)
    bind(root)
    e = CASES["C1"]
    lane = receipt(root, "talks-a")
    assert len(lane["records"]) == 1
    assert_record(lane["records"][0], e["lanes"]["talks-a"]["records"][0])
    written = root / e["lanes"]["talks-a"]["records"][0]["transcript_path"]
    assert written.read_text(encoding="utf-8") == transcript_text("1001").strip()
    assert gaps(root, "talks-a") == []
    assert registry_lanes(root) == e["registry_lanes"]


def test_c2_one_v15_talk(root: Path) -> None:
    add_v15(root, "oral-2001", kind="oral")
    add_v15_non_talk(root, "poster-2999")
    declare_sources(root, v15=True)
    bind(root)
    e = CASES["C2"]
    lane = receipt(root, "talks-b")
    assert len(lane["records"]) == 1
    assert_record(lane["records"][0], e["lanes"]["talks-b"]["records"][0])
    copies = root / "storage" / "transcription-ocr" / "evidence-lanes" / "talks-b" / "transcripts"
    assert not copies.exists() or not any(copies.iterdir())


def test_c3_a_named_exclusion_inside_a_shared_recording(root: Path) -> None:
    add_presentation(root, "3001", kind="expo_talk_panel", slideslive="5003", transcript=HALLUCINATION)
    add_presentation(root, "3002", kind="oral", slideslive="5003")
    exclude(root, "3001", "asr_hallucination")
    declare_sources(root, deep_dive=True)
    bind(root)
    e = CASES["C3"]["lanes"]["talks-a"]
    lane = receipt(root, "talks-a")
    assert len(lane["records"]) == 1
    assert_record(lane["records"][0], e["records"][0])
    assert gaps(root, "talks-a") == e["coverage_gaps"]
    assert lane["gap_sources"] == e["gap_sources"]


def test_c4_two_empty_transcripts(root: Path) -> None:
    add_presentation(root, "4001", kind="workshop", transcript=None)
    add_v15(root, "poster-4002", kind="poster", text="\n")
    declare_sources(root, deep_dive=True, v15=True)
    bind(root)
    e = CASES["C4"]
    for lane_id, expected in e["lanes"].items():
        assert receipt(root, lane_id)["records"] == expected["records"]
        assert gaps(root, lane_id) == expected["coverage_gaps"]
    assert registry_lanes(root) == e["registry_lanes"]


def _c5(root: Path) -> None:
    for event_id in ("5101", "5102", "5103"):
        add_presentation(root, event_id, kind="oral", slideslive="5100")
    declare_sources(root, deep_dive=True)


def test_c5_one_recording_three_events(root: Path) -> None:
    _c5(root)
    e = CASES["C5"]
    before = tree(root)
    dry = bind(root, dry_run=True)
    assert tree(root) == before
    assert dry.lanes["talks-a"] == {"records": e["dry_run"]["records"], "coverage_gaps": e["dry_run"]["coverage_gaps"]}
    bind(root)
    lane = receipt(root, "talks-a")
    assert len(lane["records"]) == 1
    assert_record(lane["records"][0], e["lanes"]["talks-a"]["records"][0])
    assert gaps(root, "talks-a") == e["lanes"]["talks-a"]["coverage_gaps"]


def test_c6_bind_twice_changes_nothing(root: Path) -> None:
    _c5(root)
    bind(root)
    first = {rel: digest(root / rel) for rel in CASES["C6"]["unchanged_after_second_bind"]}
    bind(root)
    assert {rel: digest(root / rel) for rel in CASES["C6"]["unchanged_after_second_bind"]} == first


def _c7(root: Path, *, talks: bool) -> None:
    paper_base(root)
    if not talks:
        return
    add_presentation(root, "7101", kind="oral", slideslive="9101")
    add_presentation(root, "7102", kind="expo_talk_panel", slideslive="9102", transcript=HALLUCINATION)
    add_presentation(root, "7103", kind="workshop", transcript=None)
    add_presentation(root, "7104", kind="keynote", slideslive="9104")
    add_presentation(root, "7105", kind="test_of_time", slideslive="9104")
    add_v15(root, "poster-7201", kind="poster")
    add_v15(root, "oral-7202", kind="oral")
    add_v15(root, "poster-7203", kind="poster", text="\n")
    exclude(root, "7102", "asr_hallucination")
    declare_sources(root, deep_dive=True, v15=True)


def test_c7_admission_with_talks(tmp_path: Path) -> None:
    with_talks, without = tmp_path / "with" / "conf", tmp_path / "without" / "conf"
    _c7(with_talks, talks=True)
    _c7(without, talks=False)
    bind(with_talks)
    admit(with_talks)
    admit(without)
    e = CASES["C7"]
    stats = read_json(with_talks / "storage" / "corpus" / "corpus_statistics.json")
    plain = read_json(without / "storage" / "corpus" / "corpus_statistics.json")
    talks = stats["talks"]
    assert {k: talks[k] for k in ("listed", "admitted", "excluded", "by_kind")} == {
        k: e["talks"][k] for k in ("listed", "admitted", "excluded", "by_kind")}
    assert [lane["lane_id"] for lane in talks["lanes"]] == e["talks"]["lanes"]
    for kind, row in talks["by_kind"].items():
        assert row["listed"] == row["admitted"] + row["excluded"], kind
    inventory = read_json(with_talks / "storage" / "corpus" / "corpus_inventory.json")
    speech = [r for r in inventory["artifact_records"] if r.get("source_family_id") == "speech_transcript"]
    assert sum(1 for r in speech if not r.get("index_exclusion")) == e["inventory"]["talk_records"]
    assert sum(1 for r in speech if r.get("index_exclusion")) == e["inventory"]["gap_records"]
    for key, value in e["inventory"]["artifact_counts"].items():
        assert inventory["artifact_counts"][key] == value
    for key in e["papers_equal_to_no_lane_admission"]:
        assert stats[key] == plain[key], key
    assert (stats["corpus_version"]["corpus_records_sha256"] != plain["corpus_version"]["corpus_records_sha256"]) is e[
        "corpus_records_sha256_differs_from_no_lane_admission"]


@pytest.mark.parametrize("case", sorted(P5_DIGESTS))
def test_c8_p5_corpus_replayed_byte_identical(tmp_path: Path, case: str) -> None:
    ws = p5.Ws(tmp_path / "conf")
    {"F0": lambda w: None, "C1": p5.c1, "C2": p5.c2, "C3": p5.c3, "C5": p5.c5}[case](ws)
    p5.run(ws)
    assert ws.outputs() == P5_DIGESTS[case]
    for name in OUTPUTS:
        assert '"talks"' not in (ws.corpus_dir() / name).read_text(encoding="utf-8")
    assert CASES["C8"]["talks_key_present"] is False


def test_c9_admission_twice_with_an_older_lane(root: Path) -> None:
    lanes = importlib.import_module("conference_pipeline.indexing.transcript_evidence_lanes")
    paper_base(root, papers=1)
    rel = "storage/transcription-ocr/evidence-lanes/legacy-captions/transcripts/yt-abc.txt"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(transcript_text("yt-abc"), encoding="utf-8")
    _, blockers = lanes.build_lane_receipt(
        workspace=root, lane_id="legacy-captions", conference="fixture", producer="fixture",
        transcript_source="youtube_caption_auto", records=[{"video_id": "yt-abc", "transcript_path": rel}],
    )
    assert blockers == []
    lanes.register_lane(workspace=root, lane_id="legacy-captions")
    add_presentation(root, "9001", kind="oral", slideslive="9001")
    declare_sources(root, deep_dive=True)
    bind(root)
    admit(root)
    corpus = root / "storage" / "corpus"
    first = {name: digest(corpus / name) for name in OUTPUTS}
    admit(root)
    e = CASES["C9"]
    assert {name: digest(corpus / name) for name in e["byte_identical_after_second_admission"]} == first
    records = read_json(corpus / "corpus_inventory.json")["artifact_records"]
    legacy = [r for r in records if r.get("video_id") == e["legacy_record"]["video_id"]]
    assert len(legacy) == 1
    assert ("item_kind" in legacy[0]) is e["legacy_record"]["has_item_kind"]
    assert legacy[0]["evidence_lane_id"] == e["legacy_record"]["evidence_lane_id"]
    talks = read_json(corpus / "corpus_statistics.json")["talks"]
    assert {k: talks[k] for k in ("listed", "admitted", "excluded", "by_kind")} == e["talks"]


def test_s1_a_missing_v15_transcript_refuses_the_bind(root: Path) -> None:
    """Successor (spec section 6.3 step 1): a transcript path that does not resolve refuses."""

    add_v15(root, "oral-1", kind="oral", write_file=False)
    declare_sources(root, v15=True)
    before = tree(root)
    with pytest.raises(lanes_module().PaperLaneError):
        bind(root)
    assert tree(root) == before


def test_s2_cli_bind_refuses_a_bad_slug(tmp_path: Path) -> None:
    cli = importlib.import_module("conference_pipeline.production.talk_admission_cli")
    messages: list[str] = []
    assert cli.bind("../x", project_root=tmp_path, echo=messages.append) == 2
    assert cli.bind("missing", project_root=tmp_path, echo=messages.append) == 2


def test_s3_cli_bind_dry_run_reports(tmp_path: Path) -> None:
    cli = importlib.import_module("conference_pipeline.production.talk_admission_cli")
    root = tmp_path / "conferences" / "conf"
    root.mkdir(parents=True)
    _c5(root)
    before = tree(root)
    messages: list[str] = []
    assert cli.bind("conf", project_root=tmp_path, dry_run=True, echo=messages.append) == 0
    assert tree(root) == before
    assert json.loads(messages[-1])["lanes"]["talks-a"] == {"records": 1, "coverage_gaps": 2}


def test_v15_paths_used_by_the_corpus_are_the_builders() -> None:
    assert v15_transcript_path("oral-2001") == CASES["C2"]["lanes"]["talks-b"]["records"][0]["transcript_path"]
    assert presentation_file("1001") in CASES["C1"]["lanes"]["talks-a"]["records"][0]["provenance"][0]


def test_s4_an_excluded_v15_transcript_is_accounted_end_to_end(root: Path) -> None:
    """Successor, finding F9: a v1.5 event that becomes a gap names its transcript file, so
    the guard matches the on-disk transcript to the gap (ICLR 2025's empty poster-29147)."""
    import os
    import subprocess
    import sys

    paper_base(root, papers=1)
    add_v15(root, "poster-1", kind="poster")
    add_v15(root, "poster-2", kind="poster", text="\n")
    declare_sources(root, v15=True)
    bind(root)
    admit(root)
    assert receipt(root, "talks-b")["gap_sources"] == {"event:poster-2": v15_transcript_path("poster-2")}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "audit_talk_coverage.py"), "--workspace", str(root)],
                          capture_output=True, text=True, env=env, check=False)
    report = json.loads(done.stdout)
    assert done.returncode == 0, report["unaccounted"]
    assert report["excluded"] == {"empty_transcript": 2}


def test_r1_f1_a_recording_id_or_lane_id_that_escapes_is_refused_before_any_write(tmp_path: Path) -> None:
    """Review phase-5.1-r1 F1: a traversal SlidesLive id or lane id never writes outside the workspace."""
    root = tmp_path / "conf"
    root.mkdir()
    add_presentation(root, "1", kind="oral", slideslive="../../../../outside")
    declare_sources(root, deep_dive=True)
    with pytest.raises(lanes_module().PaperLaneError):
        bind(root)
    assert not (tmp_path / "outside.txt").exists()
    assert not list(tmp_path.rglob("outside.txt"))
    sources = read_json(root / "storage" / "transcription-ocr" / "talk_sources.json")
    sources["sources"][0]["lane_id"] = "../escape"
    (root / "storage" / "transcription-ocr" / "talk_sources.json").write_text(json.dumps(sources), encoding="utf-8")
    with pytest.raises(lanes_module().PaperLaneError):
        bind(root)
    assert not (root / "storage" / "transcription-ocr" / "escape").exists()


def test_r1_f6_a_superseded_lane_event_stays_listed(root: Path) -> None:
    """Review phase-5.1-r1 F6: an event a later lane carries for a recording an earlier lane
    admitted is listed and excluded as superseded, never dropped from the statistics."""
    lanes = importlib.import_module("conference_pipeline.indexing.transcript_evidence_lanes")
    paper_base(root, papers=1)
    for lane_id, kind in (("first", "keynote"), ("second", "test_of_time")):
        rel = f"storage/transcription-ocr/evidence-lanes/{lane_id}/transcripts/9104.txt"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(transcript_text(lane_id), encoding="utf-8")
        _, blockers = lanes.build_lane_receipt(workspace=root, lane_id=lane_id, conference="fixture", producer="fixture",
                                               transcript_source="asr",
                                               records=[{"video_id": "9104", "transcript_path": rel, "item_kind": kind}])
        assert blockers == []
        lanes.register_lane(workspace=root, lane_id=lane_id)
    admit(root)
    talks = read_json(root / "storage" / "corpus" / "corpus_statistics.json")["talks"]
    assert (talks["listed"], talks["admitted"], talks["excluded"]) == (2, 1, {"superseded": 1})
    assert talks["by_kind"]["test_of_time"] == {"listed": 1, "admitted": 0, "excluded": 1}


def test_s5_a_deep_dive_folder_linked_into_another_store_binds(tmp_path: Path) -> None:
    """Successor, finding F14: ICLR 2025's exports/deep-dive is a link into a Drive folder.
    A linked folder is still the workspace's; binding reads through the link."""
    drive = tmp_path / "drive" / "deep-dive"
    drive.mkdir(parents=True)
    root = tmp_path / "conf"
    (root / "exports").mkdir(parents=True)
    (root / "exports" / "deep-dive").symlink_to(drive)
    add_presentation(root, "1001", kind="oral", slideslive="5001")
    declare_sources(root, deep_dive=True)
    bind(root)
    record = receipt(root, "talks-a")["records"][0]
    assert record["video_id"] == "5001"
    assert record["provenance"][0] == f"source:{presentation_file('1001')}"
