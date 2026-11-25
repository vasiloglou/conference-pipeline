"""The talk-coverage guard, corpus cases C10-C17 (v6.0-Public-Release Phase 5.1).

Each test is one case of ``tests/fixtures/talk_admission/README.md``; its answers come
from ``EXPECTED.json``. The guard runs as ``scripts/audit_talk_coverage.py`` in a
subprocess, so the exit codes are the script's own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.talk_corpus_builders import (
    V15_RECEIPT,
    add_manifest_entry_without_file,
    add_presentation,
    add_v15,
    presentation_file,
    speech_record,
    touch,
    v15_transcript_path,
    write_inventory,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_talk_coverage.py"
CASES = json.loads((REPO_ROOT / "tests" / "fixtures" / "talk_admission" / "EXPECTED.json").read_text(encoding="utf-8"))["cases"]


def guard(workspace: Path) -> tuple[int, dict[str, Any]]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, str(SCRIPT), "--workspace", str(workspace)], capture_output=True, text=True,
                          env=env, cwd=REPO_ROOT, check=False)
    return done.returncode, json.loads(done.stdout)


def check(report: dict[str, Any], code: int, expected: dict[str, Any]) -> None:
    assert code == expected["exit_code"]
    assert report["exit_code"] == expected["exit_code"]
    for key in ("on_disk", "admitted", "excluded", "unaccounted"):
        if key in expected:
            assert report[key] == expected[key], key


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "conf"
    (path / "storage").mkdir(parents=True)
    return path


def test_c10_all_admitted(root: Path) -> None:
    rel = add_v15(root, "poster-1", kind="poster")
    write_inventory(root, [speech_record(transcript_path=rel)])
    code, report = guard(root)
    check(report, code, CASES["C10"])


def test_c11_one_v15_transcript_left_out(root: Path) -> None:
    first = add_v15(root, "poster-1", kind="poster")
    add_v15(root, "poster-2", kind="poster")
    add_v15(root, "poster-3", kind="poster", write_file=False)
    write_inventory(root, [speech_record(transcript_path=first)])
    code, report = guard(root)
    check(report, code, CASES["C11"])


def test_c12_one_stage_inventory_transcript_left_out(root: Path) -> None:
    a = touch(root, "storage/transcription-ocr/speech-transcripts/speech-raw-a.txt")
    b = touch(root, "storage/transcription-ocr/speech-transcripts/speech-raw-b.txt")
    write_json(root / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json",
               {"schema_version": 1, "output_inventory": {"speech_transcript_outputs": [a, b]}})
    write_inventory(root, [speech_record(transcript_path=a)])
    code, report = guard(root)
    e = dict(CASES["C12"])
    e["unaccounted"] = [{**e["unaccounted"][0], "identity": b}]
    check(report, code, e)


def test_c12b_the_caption_shape(root: Path) -> None:
    entries, records = [], []
    for video_id in ("v1", "v2", "v3"):
        caption = touch(root, f"storage/download-corpus/event-media/captions/{video_id}.en.vtt")
        output = touch(root, f"storage/transcription-ocr/conf/caption-transcripts/{video_id}.json", "{}\n")
        entries.append({"video_id": video_id, "caption_path": caption, "output_path": output})
        if video_id == "v1":
            records.append(speech_record(transcript_path=caption, artifact_id="speech:v1"))
        if video_id == "v3":
            records.append(speech_record(transcript_path=output, artifact_id="speech:v3"))
    write_json(root / "storage" / "transcription-ocr" / "conf" / "inventory.json", {"caption_transcripts": entries})
    write_inventory(root, records)
    code, report = guard(root)
    e = dict(CASES["C12b"])
    e["unaccounted"] = [{**e["unaccounted"][0], "identity": entries[1]["caption_path"]}]
    check(report, code, e)


def test_c13_a_lane_record_missing_from_the_inventory(root: Path) -> None:
    import importlib

    lanes = importlib.import_module("conference_pipeline.indexing.transcript_evidence_lanes")
    r1 = touch(root, "storage/transcription-ocr/evidence-lanes/rec/transcripts/r1.txt")
    r2 = touch(root, "storage/transcription-ocr/evidence-lanes/rec/transcripts/r2.txt")
    _, blockers = lanes.build_lane_receipt(
        workspace=root, lane_id="rec", conference="fixture", producer="fixture", transcript_source="asr",
        records=[{"video_id": "r1", "transcript_path": r1}, {"video_id": "r2", "transcript_path": r2}],
        coverage_gaps=[{"video_id": "g1", "reason": "silent_audio"}],
    )
    assert blockers == []
    lanes.register_lane(workspace=root, lane_id="rec")
    write_inventory(root, [speech_record(transcript_path=r1)])
    code, report = guard(root)
    e = dict(CASES["C13"])
    e["unaccounted"] = [{**e["unaccounted"][0], "identity": r2}]
    check(report, code, e)


def test_c14_a_deep_dive_file_left_out(root: Path) -> None:
    f1 = add_presentation(root, "f1", kind="oral")
    add_presentation(root, "f2", kind="oral")
    add_manifest_entry_without_file(root, "f3")
    write_inventory(root, [speech_record(transcript_path="x.txt", provenance=[f"source:{f1}"])])
    code, report = guard(root)
    check(report, code, CASES["C14"])


def test_c15_exclusions_account_for_talks(root: Path) -> None:
    t1 = add_v15(root, "poster-t1", kind="poster", text="\n")
    d1 = add_presentation(root, "d1", kind="oral")
    write_inventory(root, [
        speech_record(transcript_path=t1, index_exclusion="empty_transcript", artifact_id="gap:t1"),
        speech_record(source_path=d1, index_exclusion="duplicate_recording", artifact_id="gap:d1"),
    ])
    code, report = guard(root)
    check(report, code, CASES["C15"])


def test_c16a_transcripts_but_no_inventory(root: Path) -> None:
    add_v15(root, "poster-1", kind="poster")
    code, report = guard(root)
    e = CASES["C16"]["a"]
    assert code == e["exit_code"] and report["exit_code"] == e["exit_code"]
    assert report["on_disk"] == e["on_disk"] and report["admitted"] == e["admitted"]
    assert len(report["unaccounted"]) == e["unaccounted_count"]
    assert report["reason"] == e["reason"]
    assert report["errors"] == e["errors"]


def test_c16b_an_empty_workspace(root: Path) -> None:
    code, report = guard(root)
    check(report, code, CASES["C16"]["b"])


def test_c17a_a_malformed_receipt(root: Path) -> None:
    (root / V15_RECEIPT).parent.mkdir(parents=True, exist_ok=True)
    (root / V15_RECEIPT).write_text("{not json", encoding="utf-8")
    code, report = guard(root)
    e = CASES["C17"]["a"]
    assert code == e["exit_code"] and report["exit_code"] == e["exit_code"]
    assert report["errors"] == e["errors"]
    assert report["admitted"] == e["counted_as_accounted"]


def test_c17b_a_missing_workspace(tmp_path: Path) -> None:
    code, report = guard(tmp_path / "nowhere")
    e = CASES["C17"]["b"]
    assert code == e["exit_code"] and report["exit_code"] == e["exit_code"]
    assert [error["reason"] for error in report["errors"]] == [e["errors"][0]["reason"]]


def test_builders_match_the_corpus_paths() -> None:
    assert v15_transcript_path("poster-2") == CASES["C11"]["unaccounted"][0]["identity"]
    assert f"deep_dive:{presentation_file('f2')}" == CASES["C14"]["unaccounted"][0]["identity"]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=REPO_ROOT,
                          check=False)


def test_s1_receipt_inside_the_workspace_is_written(root: Path) -> None:
    """Successor (spec section 6.6): --receipt writes the same report inside the workspace."""
    target = root / "storage" / "talk_coverage.json"
    done = _run(["--workspace", str(root), "--receipt", str(target)])
    assert done.returncode == 0
    assert json.loads(target.read_text(encoding="utf-8")) == json.loads(done.stdout)


def test_s2_receipt_outside_the_workspace_is_refused(root: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.json"
    done = _run(["--workspace", str(root), "--receipt", str(target)])
    assert done.returncode == 2
    assert not target.exists()


def test_s3_a_lane_record_superseded_by_an_earlier_lane_is_accounted(root: Path) -> None:
    """Successor, finding F7: CVPR 2026 has 13 lane records whose video an earlier lane
    admitted. ``load_admitted_lane_records`` skips them by precedence; the guard counts
    them as excluded ``superseded``, never unaccounted."""
    import importlib

    lanes = importlib.import_module("conference_pipeline.indexing.transcript_evidence_lanes")
    for lane_id in ("first", "second"):
        rel = touch(root, f"storage/transcription-ocr/evidence-lanes/{lane_id}/transcripts/v1.txt")
        _, blockers = lanes.build_lane_receipt(workspace=root, lane_id=lane_id, conference="fixture", producer="fixture",
                                               transcript_source="asr", records=[{"video_id": "v1", "transcript_path": rel}])
        assert blockers == []
        lanes.register_lane(workspace=root, lane_id=lane_id)
    record = speech_record(transcript_path="storage/transcription-ocr/evidence-lanes/first/transcripts/v1.txt")
    record["video_id"] = "v1"
    write_inventory(root, [record])
    code, report = guard(root)
    assert code == 0
    assert report["admitted"] == 1 and report["excluded"] == {"superseded": 1} and report["unaccounted"] == []


def test_s4_a_structured_exclusion_reports_its_reason(root: Path) -> None:
    """Successor, finding F6: ICML 2026 writes index_exclusion as {"reason", "evidence"}."""
    rel = add_v15(root, "poster-1", kind="poster")
    record = speech_record(transcript_path=rel, index_exclusion="x")
    record["index_exclusion"] = {"reason": "speechless_recording", "evidence": "a 4-character transcript"}
    write_inventory(root, [record])
    code, report = guard(root)
    assert code == 0
    assert report["excluded"] == {"speechless_recording": 1}


def test_s5_an_unreadable_file_is_named_as_such(root: Path) -> None:
    """Successor, finding F8: a file the process may not read (ICLR 2026's Drive-backed
    storage under macOS privacy protection) is exit 2 ``unreadable``, not ``invalid_json``."""
    rel = add_v15(root, "poster-1", kind="poster")
    receipt = root / V15_RECEIPT
    receipt.chmod(0o000)
    try:
        code, report = guard(root)
    finally:
        receipt.chmod(0o644)
    assert code == 2
    assert report["errors"] == [{"path": V15_RECEIPT.as_posix(), "reason": "unreadable"}]
    assert rel


def test_r1_f2_a_caption_output_without_its_vtt_is_still_on_disk(root: Path) -> None:
    """Review phase-5.1-r1 F2: the transcript JSON survives while the VTT is gone."""
    output = touch(root, "storage/transcription-ocr/conf/caption-transcripts/v1.json", "{}\n")
    write_json(root / "storage" / "transcription-ocr" / "conf" / "inventory.json",
               {"caption_transcripts": [{"video_id": "v1", "caption_path": "storage/missing/v1.en.vtt", "output_path": output}]})
    write_inventory(root, [])
    code, report = guard(root)
    assert code == 1 and report["on_disk"] == 1 and len(report["unaccounted"]) == 1


@pytest.mark.parametrize("receipt", ["v15", "lane"])
def test_r1_f3_a_control_file_of_the_wrong_shape_is_exit_2(root: Path, receipt: str) -> None:
    """Review phase-5.1-r1 F3: a receipt that is valid JSON but not an object is unreadable input."""
    add_v15(root, "poster-1", kind="poster")
    if receipt == "v15":
        (root / V15_RECEIPT).write_text("[]\n", encoding="utf-8")
    else:
        lane_receipt = touch(root, "storage/transcription-ocr/evidence-lanes/x/lane_receipt.json", "[]\n")
        write_json(root / "storage" / "transcription-ocr" / "evidence_lanes.json",
                   {"lanes": [{"lane_id": "x", "receipt_path": lane_receipt, "receipt_sha256": "0" * 64}]})
    write_inventory(root, [])
    code, report = guard(root)
    assert code == 2 and report["exit_code"] == 2
    assert report["errors"] and report["errors"][0]["reason"] == "invalid_structure"


def test_r1_f4_an_unreadable_transcript_is_exit_2(root: Path) -> None:
    """Review phase-5.1-r1 F4: a listed transcript the process may not read is not counted."""
    rel = add_v15(root, "poster-1", kind="poster")
    write_inventory(root, [speech_record(transcript_path=rel)])
    (root / rel).chmod(0o000)
    try:
        code, report = guard(root)
    finally:
        (root / rel).chmod(0o644)
    assert code == 2 and report["admitted"] == 0
    assert report["errors"] == [{"path": rel, "reason": "unreadable"}]


def test_r1_f5_an_exclusion_without_a_reason_does_not_account(root: Path) -> None:
    """Review phase-5.1-r1 F5: a structured exclusion with no reason names nothing."""
    rel = add_v15(root, "poster-1", kind="poster")
    record = speech_record(transcript_path=rel, index_exclusion="x")
    record["index_exclusion"] = {"evidence": "checked"}
    write_inventory(root, [record])
    code, report = guard(root)
    assert code == 1 and report["excluded"] == {} and len(report["unaccounted"]) == 1


def test_s6_a_location_that_cannot_be_listed_is_exit_2(root: Path) -> None:
    """Successor, finding F12: ICLR 2026's storage/transcription-ocr is a symlink into a Drive
    folder the process may not list. A location that exists but cannot be listed is exit 2
    ``unreadable``, never an empty workspace that passes."""
    blocked = root / "storage" / "transcription-ocr"
    blocked.mkdir(parents=True)
    (blocked / "transcription_ocr_inventory.json").write_text('{"output_inventory": {}}', encoding="utf-8")
    blocked.chmod(0o000)
    try:
        code, report = guard(root)
    finally:
        blocked.chmod(0o755)
    assert code == 2
    assert report["errors"] == [{"path": "storage/transcription-ocr", "reason": "unreadable"}]


def test_s7_a_broken_symlink_location_is_exit_2(root: Path, tmp_path: Path) -> None:
    """Successor, finding F12: a Drive folder that is not mounted leaves a dangling symlink."""
    (root / "storage" / "corpus").symlink_to(tmp_path / "not-mounted" / "corpus")
    code, report = guard(root)
    assert code == 2
    assert report["errors"] == [{"path": "storage/corpus", "reason": "unreadable"}]


def test_s8_a_linked_location_that_lists_empty_is_unverified(root: Path, tmp_path: Path) -> None:
    """Successor, finding F14: from some processes ICLR 2026's Drive-linked storage lists as
    empty rather than refusing. A linked location with nothing in it is exit 2 ``unverified``,
    never an empty workspace that passes."""
    empty = tmp_path / "drive" / "transcription-ocr"
    empty.mkdir(parents=True)
    (root / "storage" / "transcription-ocr").symlink_to(empty)
    code, report = guard(root)
    assert code == 2
    assert report["errors"] == [{"path": "storage/transcription-ocr", "reason": "unverified_empty_link"}]


def test_s9_a_linked_location_with_files_is_read(root: Path, tmp_path: Path) -> None:
    """The F14 rule leaves a populated linked location alone (NeurIPS-style Drive links that work)."""
    target = tmp_path / "drive" / "transcription-ocr"
    target.mkdir(parents=True)
    (root / "storage" / "transcription-ocr").symlink_to(target)
    rel = add_v15(root, "poster-1", kind="poster")
    write_inventory(root, [speech_record(transcript_path=rel)])
    code, report = guard(root)
    assert code == 0 and report["admitted"] == 1


def _guard_in(cwd: Path, workspace_arg: str) -> tuple[int, dict[str, Any]]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, str(SCRIPT), "--workspace", workspace_arg], capture_output=True, text=True,
                          env=env, cwd=cwd, check=False)
    return done.returncode, json.loads(done.stdout)


def test_r2_f1_a_relative_workspace_path_counts_the_same(root: Path, tmp_path: Path) -> None:
    """Review phase-5.1-r2 F1: --workspace given relative to the working directory."""
    add_presentation(root, "f1", kind="oral")
    write_inventory(root, [])
    absolute = guard(root)
    relative = _guard_in(tmp_path, "conf")
    assert absolute[1]["on_disk"] == relative[1]["on_disk"] == 1
    assert absolute[0] == relative[0] == 1


def test_r2_f2_a_nested_folder_that_cannot_be_listed_is_exit_2(root: Path) -> None:
    """Review phase-5.1-r2 F2: storage/transcription-ocr/<conf>/ hides a stage inventory."""
    nested = root / "storage" / "transcription-ocr" / "conf"
    nested.mkdir(parents=True)
    write_json(nested / "inventory.json", {"caption_transcripts": []})
    nested.chmod(0o000)
    try:
        code, report = guard(root)
    finally:
        nested.chmod(0o755)
    assert code == 2
    assert report["errors"] == [{"path": "storage/transcription-ocr/conf", "reason": "unreadable"}]


def test_r2_f3_a_lane_receipt_outside_the_workspace_is_exit_2(root: Path, tmp_path: Path) -> None:
    """Review phase-5.1-r2 F3: the registry names a receipt by '..'."""
    outside = tmp_path / "other" / "lane_receipt.json"
    write_json(outside, {"records": [], "coverage_gaps": [{"video_id": "x", "reason": "silent_audio"}]})
    write_json(root / "storage" / "transcription-ocr" / "evidence_lanes.json",
               {"lanes": [{"lane_id": "x", "receipt_path": "../other/lane_receipt.json", "receipt_sha256": "0" * 64}]})
    write_inventory(root, [])
    code, report = guard(root)
    assert code == 2
    assert report["errors"] == [{"path": "../other/lane_receipt.json", "reason": "invalid_structure"}]


def test_r2_f4_an_empty_exclusion_object_does_not_admit(root: Path) -> None:
    """Review phase-5.1-r2 F4: index_exclusion {} is present but names no reason."""
    rel = add_v15(root, "poster-1", kind="poster")
    record = speech_record(transcript_path=rel)
    record["index_exclusion"] = {}
    write_inventory(root, [record])
    code, report = guard(root)
    assert code == 1 and report["admitted"] == 0 and len(report["unaccounted"]) == 1


@pytest.mark.parametrize("kind", ["empty_link", "broken_link"])
def test_r3_f1_a_nested_linked_folder_never_vanishes(root: Path, tmp_path: Path, kind: str) -> None:
    """Review phase-5.1-r3 F1: storage/transcription-ocr/<conf>/ is a link that lists empty or is broken."""
    (root / "storage" / "transcription-ocr").mkdir(parents=True)
    target = tmp_path / "drive" / "conf"
    if kind == "empty_link":
        target.mkdir(parents=True)
    (root / "storage" / "transcription-ocr" / "conf").symlink_to(target)
    write_inventory(root, [])
    code, report = guard(root)
    assert code == 2
    assert report["errors"][0]["path"] == "storage/transcription-ocr/conf"


def test_r3_f2_a_control_file_that_is_a_broken_link_is_exit_2(root: Path, tmp_path: Path) -> None:
    """Review phase-5.1-r3 F2: transcription_ocr_inventory.json links to a file that is not there."""
    (root / "storage" / "transcription-ocr").mkdir(parents=True)
    (root / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json").symlink_to(tmp_path / "gone.json")
    write_inventory(root, [])
    code, report = guard(root)
    assert code == 2
    assert report["errors"] == [{"path": "storage/transcription-ocr/transcription_ocr_inventory.json", "reason": "unreadable"}]


def test_r3_f2_a_transcript_whose_folder_cannot_be_searched_is_exit_2(root: Path) -> None:
    """Review phase-5.1-r3 F2: stat of a listed transcript is refused; it is not absent."""
    rel = add_v15(root, "poster-1", kind="poster")
    write_inventory(root, [speech_record(transcript_path=rel)])
    folder = (root / rel).parent
    folder.chmod(0o000)
    try:
        code, report = guard(root)
    finally:
        folder.chmod(0o755)
    assert code == 2
    assert report["errors"] == [{"path": rel, "reason": "unreadable"}]
