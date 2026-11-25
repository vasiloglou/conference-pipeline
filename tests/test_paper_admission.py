"""The paper-admission corpus (v6.0-Public-Release Phase 5).

Each test is one case of ``tests/fixtures/paper_admission/README.md``; its numbers come
from the answer key ``EXPECTED.json``. Cases run in ``tmp_path``; no network, no real
workspace.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "tests" / "fixtures" / "paper_admission"
EXPECTED = json.loads((CORPUS / "EXPECTED.json").read_text(encoding="utf-8"))
CASES = EXPECTED["cases"]

FULL = "Full text. " + ("This paragraph stands in for the body of a paper. " * 40)
SHORT = "A short note, well under the full-text threshold."

F0_PAPERS = [
    {"id": "AAAAAAAAA1", "forum": "AAAAAAAAA1", "title": "Sparse Mixtures for Long Context Reasoning", "authors": ["Ada One", "Ben Two"], "abstract": "Abstract one.", "track": "main"},
    {"id": "AAAAAAAAA2", "forum": "AAAAAAAAA2", "title": "Calibrated Uncertainty in Diffusion Planners", "authors": ["Cy Three"], "abstract": "Abstract two.", "track": "main"},
    {"id": "AAAAAAAAA3", "forum": "AAAAAAAAA3", "title": "Graph Priors for Molecular Generation Tasks", "authors": ["Di Four"], "abstract": "Abstract three.", "track": "main"},
    {"id": "BBBBBBBBB1", "forum": "BBBBBBBBB1", "title": "Benchmarks for Scientific Agents in the Lab", "authors": ["Ed Five"], "abstract": "Abstract four.", "track": "workshop"},
]


def admission():
    return importlib.import_module("conference_pipeline.production.paper_admission")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class Ws:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lanes: list[dict[str, Any]] = []
        _write(root / "storage" / "acquisition" / "papers.json", {"papers": F0_PAPERS})
        self._registry()

    def _registry(self) -> None:
        _write(
            self.root / "storage" / "papers" / "late_paper_lanes.json",
            {
                "schema_version": 1,
                "existing_corpus": {"adapter": "openreview_acquisition_json", "path": "storage/acquisition/papers.json"},
                "lanes": self.lanes,
            },
        )

    def lane(self, lane_id: str, *, track: str, id_kind: str, units: list[dict[str, Any]]) -> None:
        rows = []
        for unit in units:
            row = {"unit_id": unit["unit_id"], "title": unit["title"], "kind": "paper_text"}
            text_path = self.root / "storage" / "papers" / "late" / lane_id / "text" / f"{len(rows)}.txt"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(unit["text"], encoding="utf-8")
            row["path"] = text_path.relative_to(self.root).as_posix()
            rows.append(row)
        unit_file = self.root / "storage" / "papers" / "late" / lane_id / "units.jsonl"
        unit_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.lanes.append(
            {
                "lane_id": lane_id,
                "unit_file": unit_file.relative_to(self.root).as_posix(),
                "track": track,
                "id_kind": id_kind,
                "provenance": {"source": "fixture", "match_basis": "stable_id", "auth_mode": "none"},
            }
        )
        self._registry()

    def exclude(self, paper_id: str, reason: str) -> None:
        _write(self.root / "storage" / "papers" / "paper_exclusions.json", {"exclusions": [{"id": paper_id, "reason": reason}]})

    def corpus_dir(self) -> Path:
        return self.root / "storage" / "corpus"

    def outputs(self) -> dict[str, str]:
        return {
            name: hashlib.sha256((self.corpus_dir() / name).read_bytes()).hexdigest()
            for name in ("corpus_inventory.json", "paper_identity_report.json", "corpus_statistics.json")
        }

    def inventory(self) -> list[dict[str, Any]]:
        return json.loads((self.corpus_dir() / "corpus_inventory.json").read_text(encoding="utf-8"))["artifact_records"]


@pytest.fixture
def ws(tmp_path: Path) -> Ws:
    return Ws(tmp_path / "conf")


def c1(ws: Ws) -> None:
    ws.lane("ws_text", track="workshop", id_kind="openreview", units=[{"unit_id": "NNNNNNNNN1", "title": "Active Learning for Protein Stability Screens", "text": FULL}])


def c2(ws: Ws) -> None:
    ws.lane("main_text", track="main", id_kind="openreview", units=[{"unit_id": "AAAAAAAAA1", "title": "Sparse Mixtures for Long Context Reasoning", "text": FULL}])


def c3(ws: Ws) -> None:
    title = "Neural Operators for Ocean Forecasting"
    ws.lane("arxiv_a", track="workshop", id_kind="arxiv", units=[{"unit_id": "2501.00001v1", "title": title, "text": FULL}])
    ws.lane("arxiv_b", track="workshop", id_kind="arxiv", units=[{"unit_id": "2501.00001v2", "title": title, "text": FULL + " v2"}])


def c5(ws: Ws) -> None:
    ws.exclude("openreview:BBBBBBBBB1", "rejected_submission")


def run(ws: Ws, **kwargs: Any):
    return admission().admit_paper_sets(ws.root, **kwargs)


def test_c1_a_genuinely_new_paper(ws: Ws) -> None:
    c1(ws)
    r = run(ws)
    e = CASES["C1"]
    assert r.papers_total == e["papers_total"]
    assert list(r.new_paper_keys) == e["new_paper_keys"]
    assert list(r.text_gained_keys) == e["text_gained_keys"]
    assert r.full_text_after == e["full_text_after"]
    assert r.by_track["workshop"] == 2


def test_c2_an_existing_paper_gains_text(ws: Ws) -> None:
    c2(ws)
    r = run(ws)
    e = CASES["C2"]
    assert r.papers_total == e["papers_total"]
    assert list(r.new_paper_keys) == e["new_paper_keys"]
    assert list(r.text_gained_keys) == e["text_gained_keys"]
    assert r.full_text_after == e["full_text_after"]
    keys = [rec["paper_key"] for rec in ws.inventory()]
    assert keys.count("openreview:AAAAAAAAA1") == 2


def test_c3_one_paper_through_two_lanes(ws: Ws) -> None:
    c3(ws)
    r = run(ws)
    e = CASES["C3"]
    assert r.papers_total == e["papers_total"]
    assert list(r.new_paper_keys) == e["new_paper_keys"]
    assert r.duplicate_copies_merged == e["duplicate_copies_merged"]
    assert r.full_text_after == e["full_text_after"]


def test_c4_a_site_only_copy_joins_by_title(ws: Ws) -> None:
    ws.lane("site", track="workshop", id_kind="site", units=[{"unit_id": "site_abc123", "title": "Calibrated uncertainty in diffusion planners.", "text": FULL}])
    r = run(ws)
    e = CASES["C4"]
    assert r.papers_total == e["papers_total"]
    assert list(r.new_paper_keys) == e["new_paper_keys"]
    assert list(r.text_gained_keys) == e["text_gained_keys"]
    assert [dict(j) for j in r.title_joins] == e["title_joins"]
    report = json.loads((ws.corpus_dir() / "paper_identity_report.json").read_text(encoding="utf-8"))
    assert report["title_joins"] == e["title_joins"]


def test_c4b_two_stable_ids_with_one_title_stay_two(ws: Ws) -> None:
    ws.lane("ws_text", track="workshop", id_kind="openreview", units=[{"unit_id": "NNNNNNNNN2", "title": "Graph Priors for Molecular Generation Tasks", "text": FULL}])
    r = run(ws)
    e = CASES["C4b"]
    assert r.papers_total == e["papers_total"]
    assert list(r.new_paper_keys) == e["new_paper_keys"]
    assert [dict(j) for j in r.title_joins] == e["title_joins"]
    assert r.title_collisions, "the shared title is listed as a collision"


def test_c5_a_named_exclusion_is_kept_not_counted(ws: Ws) -> None:
    c5(ws)
    r = run(ws)
    e = CASES["C5"]
    assert r.papers_total == e["papers_total"]
    assert [dict(x) for x in r.exclusions] == e["exclusions"]
    w1 = [rec for rec in ws.inventory() if rec["paper_key"] == "openreview:BBBBBBBBB1"]
    assert w1 and all(rec.get("index_exclusion") == "rejected_submission" for rec in w1)
    assert r.by_track.get("workshop", 0) == 0


def test_c6_an_existing_paper_without_a_late_counterpart_is_unchanged(ws: Ws) -> None:
    c2(ws)
    r = run(ws)
    assert r.papers_total == CASES["C6"]["papers_total"]
    p3 = [rec for rec in ws.inventory() if rec["paper_key"] == "openreview:AAAAAAAAA3"]
    assert len(p3) == 1
    assert p3[0]["paper_title"] == "Graph Priors for Molecular Generation Tasks"
    assert p3[0]["authors"] == ["Di Four"]
    assert p3[0]["track"] == "main"
    assert p3[0]["metadata"]["content"] == "abstract"
    assert "index_exclusion" not in p3[0]
    # Its text is unchanged: the record's digest is the F0 abstract's (review F5).
    assert p3[0]["source_sha256"] == hashlib.sha256("Abstract three.".encode("utf-8")).hexdigest()
    assert p3[0]["source_byte_count"] == len("Abstract three.")


def test_c7_a_short_text_is_an_artifact_not_text_coverage(ws: Ws) -> None:
    ws.lane("main_text", track="main", id_kind="openreview", units=[{"unit_id": "AAAAAAAAA2", "title": "Calibrated Uncertainty in Diffusion Planners", "text": SHORT}])
    r = run(ws)
    e = CASES["C7"]
    assert r.papers_total == e["papers_total"]
    assert list(r.text_gained_keys) == e["text_gained_keys"]
    assert r.full_text_after == e["full_text_after"]
    assert r.artifacts_total == e["artifacts_total"]


def test_c8_admission_twice_changes_nothing(ws: Ws) -> None:
    c3(ws)
    run(ws)
    first = ws.outputs()
    run(ws)
    assert ws.outputs() == first


def test_c9_a_dry_run_writes_nothing(ws: Ws) -> None:
    c1(ws)
    r = run(ws, dry_run=True)
    assert r.papers_total == CASES["C9"]["papers_total"]
    assert r.written is False
    assert not ws.corpus_dir().exists()


def test_c10_statistics_equal_a_recount(ws: Ws) -> None:
    c1(ws)
    c2(ws)
    c3(ws)
    c5(ws)
    run(ws)
    e = CASES["C10"]
    stats = json.loads((ws.corpus_dir() / "corpus_statistics.json").read_text(encoding="utf-8"))
    assert stats["papers"]["total"] == e["papers_total"]
    assert stats["papers"]["by_track"] == e["by_track"]
    assert stats["text_coverage"]["full_text_after"] == e["full_text_after"]
    assert stats["text_coverage"]["gained"] == e["text_gained"]
    assert stats["artifacts"]["duplicate_copies_merged"] == e["duplicate_copies_merged"]
    assert stats["previous_corpus"]["papers_total"] == e["previous_corpus_papers_total"]
    assert stats["text_coverage"]["full_text_before"] == EXPECTED["base"]["existing_full_text"]
    # The corpus version is the digest of the inventory's records, recomputed (review F5).
    records_for_digest = json.loads((ws.corpus_dir() / "corpus_inventory.json").read_text(encoding="utf-8"))["artifact_records"]
    digest = hashlib.sha256(json.dumps(records_for_digest, sort_keys=True).encode("utf-8")).hexdigest()
    assert stats["corpus_version"]["corpus_records_sha256"] == digest
    assert stats["corpus_version"]["label"] == f"{ws.root.name}-corpus-{digest[:12]}"
    # The recount: every number from the inventory itself.
    records = ws.inventory()
    counted = {rec["paper_key"] for rec in records if not rec.get("index_exclusion")}
    assert len(counted) == stats["papers"]["total"]
    tracks: dict[str, int] = {}
    for rec in records:
        if rec.get("index_exclusion") is None and rec.get("kept_copy"):
            tracks[rec["paper_track"]] = tracks.get(rec["paper_track"], 0) + 1
    assert tracks == stats["papers"]["by_track"]
    full = {rec["paper_key"] for rec in records if not rec.get("index_exclusion") and rec["metadata"]["content"] == "full_text"}
    full |= {rec["paper_key"] for rec in records if rec.get("index_exclusion") == "duplicate_copy" and rec["metadata"]["content"] == "full_text"}
    assert len(full) == stats["text_coverage"]["full_text_after"]
    assert len(records) == stats["artifacts"]["total"]
    gained_recount = {
        rec["paper_key"] for rec in records
        if rec["metadata"]["content"] == "full_text" and rec["metadata"]["origin"] != "existing"
        and rec.get("index_exclusion") in (None, "duplicate_copy")
    } & {rec["paper_key"] for rec in records if rec["metadata"]["origin"] == "existing"}
    assert len(gained_recount) == stats["text_coverage"]["gained"]


def test_audit_reads_every_paper_record() -> None:
    """IF1: every admitted paper record is classified as a paper family by the audit."""
    from conference_pipeline.indexing.paper_source_metadata import classify_record_family

    assert classify_record_family({"source_family_id": "paper_text"}) == "papers_main"
    assert classify_record_family({"source_family_id": "workshop_paper_text"}) == "papers_workshop"


def test_cli_admit_refuses_without_a_registry(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from conference_pipeline.cli import app

    (tmp_path / "conferences" / "emptyconf").mkdir(parents=True)
    result = CliRunner().invoke(app, ["papers", "admit", "emptyconf", "--project-root", str(tmp_path)])
    assert result.exit_code == 2
    assert "late_paper_lanes.json" in result.output


def test_cli_admit_dry_run_reports_and_writes_nothing(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from conference_pipeline.cli import app

    ws = Ws(tmp_path / "conferences" / "fixtureconf")
    c1(ws)
    result = CliRunner().invoke(app, ["papers", "admit", "fixtureconf", "--project-root", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["papers_total"] == 5
    assert not ws.corpus_dir().exists()


# --- successors (implementation findings, not owner-approved cases) ------------------


def test_s1_auto_ids_track_map_and_arxiv_urls(ws: Ws) -> None:
    """IF4: auto id kind, a lane track map, and an arXiv key from the unit URL."""
    unit_file = ws.root / "storage" / "papers" / "late" / "mixed" / "units.jsonl"
    unit_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"unit_id": "AAAAAAAAA3", "title": "Graph Priors for Molecular Generation Tasks", "text": FULL, "track": "Conference"},
        {"unit_id": "prog-ws-17", "title": "Neural Operators for Ocean Forecasting", "text": FULL, "url": "https://arxiv.org/abs/2501.00001v4"},
        {"unit_id": "XXXXXXXXX9", "title": "Neural Operators for Ocean Forecasting", "text": FULL, "url": "https://export.arxiv.org/pdf/2501.00001"},
    ]
    unit_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ws.lanes.append({"lane_id": "mixed", "unit_file": unit_file.relative_to(ws.root).as_posix(), "track": "workshop",
                     "id_kind": "auto", "track_map": {"Conference": "main"}, "provenance": {"source": "fixture"}})
    ws._registry()
    r = run(ws)
    assert r.papers_total == 5  # F0's four, plus one paper reached by two ids sharing an arXiv key
    assert r.new_paper_keys == ("openreview:XXXXXXXXX9",)  # the OpenReview id names the paper (IF9)
    assert "openreview:AAAAAAAAA3" in r.text_gained_keys
    p3 = [rec for rec in ws.inventory() if rec["paper_key"] == "openreview:AAAAAAAAA3" and rec["metadata"]["origin"] == "mixed"]
    assert p3[0]["track"] == "main"


def test_s2_a_raw_line_separator_inside_a_unit_does_not_split_it(ws: Ws) -> None:
    """IF7: the lane reader splits records on newline only."""
    unit_file = ws.root / "storage" / "papers" / "late" / "raw" / "units.jsonl"
    unit_file.parent.mkdir(parents=True, exist_ok=True)
    unit_file.write_text(json.dumps({"unit_id": "NNNNNNNNN7", "title": "T", "text": "a b"}, ensure_ascii=False) + "\n", encoding="utf-8")
    ws.lanes.append({"lane_id": "raw", "unit_file": unit_file.relative_to(ws.root).as_posix(), "track": "workshop", "id_kind": "openreview", "content": "abstract"})
    ws._registry()
    assert run(ws).papers_total == 5


def test_s3_authors_from_a_papers_keyed_receipt(ws: Ws) -> None:
    """IF8: a lookup receipt {"papers": [{"key", "authors"}]} supplies a lane's authors."""
    ws.lane("arx", track="workshop", id_kind="site", units=[{"unit_id": "ws-sched-1", "title": "Some Unseen Workshop Paper Title", "text": FULL}])
    receipt = ws.root / "storage" / "papers" / "late" / "arx" / "receipt.json"
    receipt.write_text(json.dumps({"papers": [{"key": "ws-sched-1", "authors": ["Fay Six", "Gil Seven"]}]}), encoding="utf-8")
    ws.lanes[-1]["authors_file"] = receipt.relative_to(ws.root).as_posix()
    ws._registry()
    run(ws)
    (rec,) = [r for r in ws.inventory() if r["metadata"]["origin"] == "arx"]
    assert rec["authors"] == ["Fay Six", "Gil Seven"]


def test_s4_author_names_never_carry_an_email_address(ws: Ws) -> None:
    """IF11: an address printed beside an author name is dropped from the inventory."""
    unit_file = ws.root / "storage" / "papers" / "late" / "site" / "units.jsonl"
    unit_file.parent.mkdir(parents=True, exist_ok=True)
    unit = {"unit_id": "site_x", "title": "A Workshop Paper With Many Authors Listed", "text": FULL,
            "authors": ["Hal Eight (hal@university.example)", "someone@lab.example", "Ivy Nine"]}
    unit_file.write_text(json.dumps(unit) + "\n", encoding="utf-8")
    ws.lanes.append({"lane_id": "site", "unit_file": unit_file.relative_to(ws.root).as_posix(), "track": "workshop", "id_kind": "site"})
    ws._registry()
    run(ws)
    (rec,) = [r for r in ws.inventory() if r["metadata"]["origin"] == "site"]
    assert rec["authors"] == ["Hal Eight", "Ivy Nine"]
    assert "@" not in (ws.corpus_dir() / "corpus_inventory.json").read_text(encoding="utf-8")


def test_r1_f4_a_repeated_unit_id_gets_its_own_artifact_id(ws: Ws) -> None:
    """F4: two units with one id in one lane are two artifacts of one paper, with distinct artifact ids."""
    ws.lane("site", track="workshop", id_kind="site",
            units=[{"unit_id": "site_dup", "title": "A Duplicated Site Listing Title", "text": FULL},
                   {"unit_id": "site_dup", "title": "A Duplicated Site Listing Title", "text": FULL + " second copy"}])
    run(ws)
    recs = [r for r in ws.inventory() if r["metadata"]["origin"] == "site"]
    assert len(recs) == 2
    assert len({r["artifact_id"] for r in recs}) == 2
    assert len({r["paper_key"] for r in recs}) == 1
    ids = [r["artifact_id"] for r in ws.inventory()]
    assert len(ids) == len(set(ids))
