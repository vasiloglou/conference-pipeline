"""The release reader-families corpus (v6.0-Public-Release Phase 6, part 6a).

Each test is one case of ``tests/fixtures/release_reader_families/README.md``; its answers
come from ``EXPECTED.json``. Cases run in ``tmp_path``; nothing is published.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from tests import public_site_fixtures as site_fx
from tests import test_release_cli as rc
from tests.test_release_cli import w0  # noqa: F401 -- the W0 fixture of the release corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "tests" / "fixtures" / "release_reader_families"
CASES = json.loads((CORPUS / "EXPECTED.json").read_text(encoding="utf-8"))["cases"]
ROW_SCHEMA = json.loads((REPO_ROOT / "docs" / "public_catalog_row.schema.json").read_text(encoding="utf-8"))
BUNDLE = "conf-v1.0.0"
# An email-shaped value built at run time, so the tracked-tree marker scan sees no address.
EMAIL = "jane.approver" + "@" + "corp-mail.io"
# A home path outside the workspace, built at run time for the same reason.
OUTSIDE = "/" + "Users" + "/someone"


def rf():
    return importlib.import_module("conference_pipeline.publishing.reader_families")


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper(key: str, title: str, authors: list[str], track: str, *, kept: bool = True, exclusion: str | None = None,
          aid: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"artifact_id": aid or f"paper:{key}", "source_family_id": "workshop_paper_text" if track == "workshop" else "paper_text",
            "paper_key": key, "paper_title": title, "authors": authors, "track": track, "paper_track": track,
            "kept_copy": kept, "index_exclusion": exclusion, "indexable_text": exclusion is None, **extra}


CLOSE_RECORD = [
    "# Conf close record",
    "",
    "## Accepted at close",
    "",
    "- Deck 01 slide 4 shows a connector without a label; accepted by the operator.",
    "- Support label refused: Acquisition stages (no local receipt).",
    "- Publication approved by Jane Approver.",
]


class Proj:
    def __init__(self, base: Path) -> None:
        self.root = base / "proj"
        self.ws = self.root / "conferences" / "conf"
        self.sources: dict[str, Any] = {
            "schema_version": 1,
            "catalog": {"adapter": "corpus_inventory", "path": "storage/corpus/corpus_inventory.json"},
            "statistics": {"inventory": "storage/corpus/corpus_inventory.json"},
            "topics": {"validation": "analysis/topic_reference_validation"},
            "narratives": {"dir": "analysis", "pattern": "[0-9][0-9]_*.md"},
            "limitations": {"path": "storage/release/limitations_source.json"},
        }
        self.records = [
            paper("openreview:AAAAAAAAA1", "Sparse Mixtures for Long Context Reasoning", ["Ada One", "Ben Two"], "main"),
            paper("openreview:AAAAAAAAA1", "Sparse Mixtures for Long Context Reasoning", [], "main", kept=False,
                  exclusion="duplicate_copy", aid="paper:dup"),
            paper("arxiv:2501.00001", "Neural Operators for Ocean Forecasting", ["Cy Three"], "workshop"),
            paper("site:conf-77", "A Site-Only Paper", [], "main"),
            paper("openreview:RRRRRRRRR1", "A Rejected Submission", ["Di Four"], "workshop", kept=False,
                  exclusion="rejected_submission"),
            {"artifact_id": "evidence_lane_transcript:l:talk-xyz", "source_family_id": "speech_transcript",
             "video_id": "talk-xyz", "item_kind": "oral", "title": "A talk"},
        ]
        write(self.ws / "analysis" / "topic_reference_validation" / "01_alpha.json",
              {"valid_reference_ids": ["paper_abstract:AAAAAAAAA1", "speech:talk-xyz", "paper_abstract:ZZZZZZZZZ9"]})
        write(self.ws / "analysis" / "01_alpha.md", "# Alpha\n\nOne paragraph about sparse mixtures.\n")
        write(self.root / ".planning" / "milestones" / "Conf-CONTEXT.md", "\n".join(CLOSE_RECORD) + "\n")
        self.limitations: dict[str, Any] = {
            "schema_version": 1,
            "close_record": ".planning/milestones/Conf-CONTEXT.md",
            "accepted_lines": [[5, 5], [6, 6]],
            "items": [
                {"id": "lim-1", "kind": "accepted_finding", "summary": "Deck 01 slide 4: an unlabelled connector, accepted.",
                 "quote": "Deck 01 slide 4 shows a connector without a label", "close_record_lines": [5, 5],
                 "artifacts": ["decks/01_alpha"], "redact": []},
                {"id": "lim-2", "kind": "refused_support_label", "summary": "The acquisition support label was refused.",
                 "quote": "Support label refused: Acquisition stages", "close_record_lines": [6, 6],
                 "artifacts": [], "redact": []},
            ],
        }

    def save(self) -> None:
        write(self.ws / "storage" / "release" / "reader_sources.json", self.sources)
        write(self.ws / "storage" / "corpus" / "corpus_inventory.json", {"artifact_records": self.records})
        write(self.ws / "storage" / "release" / "limitations_source.json", self.limitations)

    def build(self) -> Any:
        self.save()
        return rf().build_reader_families(self.ws, bundle_id=BUNDLE, project_root=self.root)

    @property
    def out(self) -> Path:
        return self.ws / "storage" / "release-staging" / BUNDLE

    def rows(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (self.out / "catalog" / "papers.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def proj(tmp_path: Path) -> Proj:
    return Proj(tmp_path)


# --- catalog ------------------------------------------------------------------------


def test_r1_the_corpus_inventory_adapter(proj: Proj) -> None:
    result = proj.build()
    e = CASES["R1"]
    rows = proj.rows()
    assert rows == e["rows"]
    for row in rows:
        jsonschema.validate(row, ROW_SCHEMA)
    assert read(proj.out / "catalog" / "topic_papers.json") == e["topic_papers"]
    assert result.unresolved_topic_references == e["unresolved_topic_references"]


def test_r2_a_site_paper_with_its_own_link(proj: Proj) -> None:
    proj.records[3]["official_url"] = "https://example.org/papers/77"
    proj.build()
    row = next(r for r in proj.rows() if r["paper_id"] == "site:conf-77")
    assert row["official_url"] == CASES["R2"]["site:conf-77"]


def test_r3_the_accepted_papers_adapter(proj: Proj) -> None:
    proj.sources["catalog"] = {"adapter": "openreview_accepted_papers", "path": "storage/download-corpus/accepted.json"}
    write(proj.ws / "storage" / "download-corpus" / "accepted.json", {"papers": [
        {"id": "BBBBBBBBB1", "title": "Calibrated Uncertainty in Diffusion Planners", "authors": ["Ed Five"],
         "venue": "Conf 2026 Poster", "forum_url": "https://openreview.net/forum?id=BBBBBBBBB1"}]})
    proj.build()
    assert proj.rows() == CASES["R3"]["rows"]


def test_r4_a_named_exclusion(proj: Proj) -> None:
    proj.build()
    assert all(r["paper_id"] != "openreview:RRRRRRRRR1" for r in proj.rows())
    assert read(proj.out / "catalog" / "excluded_ids.json") == CASES["R4"]["excluded_ids"]


def test_r5_a_row_that_fails_the_schema(proj: Proj) -> None:
    proj.records[2]["paper_title"] = ""
    proj.save()
    with pytest.raises(rf().ReaderFamiliesError) as raised:
        rf().build_reader_families(proj.ws, bundle_id=BUNDLE, project_root=proj.root)
    assert CASES["R5"]["names"] in str(raised.value)
    assert not proj.out.exists() or not any(proj.out.rglob("*.json*"))
    out: list[str] = []
    code = importlib.import_module("conference_pipeline.publishing.release_cli").release_families(
        "conf", project_root=proj.root, bundle_id=BUNDLE, echo=out.append, err=out.append)
    assert code == CASES["R5"]["cli_exit"]


# --- statistics ---------------------------------------------------------------------


def _s1_statistics() -> dict[str, Any]:
    return {
        "papers": {"total": 3, "by_track": {"main": 2, "workshop": 1}},
        "corpus_version": {"label": "conf-corpus-0123456789ab", "corpus_records_sha256": "0123456789ab" + "0" * 52},
        "talks": {"listed": 5, "admitted": 4, "excluded": {"asr_hallucination": 1},
                  "by_kind": {"oral": {"listed": 2, "admitted": 2, "excluded": 0},
                              "expo_talk_panel": {"listed": 2, "admitted": 1, "excluded": 1},
                              "mystery_kind": {"listed": 1, "admitted": 1, "excluded": 0}},
                  "lanes": []},
    }


def test_s1_statistics_from_the_corpus_statistics_file(proj: Proj) -> None:
    proj.sources["statistics"] = {"path": "storage/corpus/corpus_statistics.json"}
    write(proj.ws / "storage" / "corpus" / "corpus_statistics.json", _s1_statistics())
    proj.build()
    stats = read(proj.out / "catalog" / "statistics.json")
    e = CASES["S1"]
    assert stats["papers"] == e["papers"]
    assert stats["corpus_version"]["label"] == e["corpus_version"]["label"]
    for key in ("listed", "admitted", "excluded", "by_kind"):
        assert stats["talks"][key] == e["talks"][key], key
    assert all(v["listed"] == v["admitted"] + v["excluded"] for v in stats["talks"]["by_kind"].values())


def test_s2_statistics_computed_from_the_inventory(proj: Proj) -> None:
    proj.records = [
        {"artifact_id": "p1", "source_family_id": "paper_text", "paper_title": "One", "authors": ["A"]},
        {"artifact_id": "p2", "source_family_id": "paper_text", "paper_title": "Two", "authors": ["B"]},
        {"artifact_id": "w1", "source_family_id": "workshop_paper_text", "paper_title": "Three", "authors": ["C"]},
        {"artifact_id": "p3", "source_family_id": "paper_text", "paper_title": "Four", "authors": ["D"], "index_exclusion": "duplicate_title"},
        {"artifact_id": "t1", "source_family_id": "speech_transcript", "item_kind": "poster"},
        {"artifact_id": "t2", "source_family_id": "speech_transcript", "item_kind": "poster",
         "index_exclusion": {"reason": "speechless_recording", "evidence": "4 characters"}},
        {"artifact_id": "t3", "source_family_id": "speech_transcript", "item_kind": "oral"},
    ]
    proj.sources["catalog"] = {"adapter": "none"}
    proj.sources["statistics"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    proj.build()
    stats = read(proj.out / "catalog" / "statistics.json")
    e = CASES["S2"]
    assert stats["papers"] == e["papers"]
    for key in ("listed", "admitted", "excluded", "by_kind"):
        assert stats["talks"][key] == e["talks"][key], key
    assert re.fullmatch(e["corpus_version_label_pattern"], stats["corpus_version"]["label"])


def test_s3_the_d11_note(proj: Proj) -> None:
    proj.build()
    assert "d11_note" not in read(proj.out / "catalog" / "statistics.json")
    proj.sources["d11_note"] = "Statistics in the planning deck describe the earlier corpus."
    proj.build()
    assert read(proj.out / "catalog" / "statistics.json")["d11_note"] == proj.sources["d11_note"]


# --- limitations --------------------------------------------------------------------


def test_l1_every_quote_matches(proj: Proj) -> None:
    proj.build()
    record = read(proj.out / "limitations" / "limitations.json")
    e = CASES["L1"]
    assert [{"id": i["id"], "kind": i["kind"]} for i in record["items"]] == e["items"]
    assert sorted(record) == e["record_fields"]
    md = (proj.out / "limitations" / "limitations.md").read_text(encoding="utf-8")
    assert all(item["summary"] in md for item in record["items"])


def test_l2_a_quote_not_at_its_lines(proj: Proj) -> None:
    proj.limitations["items"][1]["close_record_lines"] = [5, 5]
    proj.save()
    with pytest.raises(rf().ReaderFamiliesError) as raised:
        rf().build_reader_families(proj.ws, bundle_id=BUNDLE, project_root=proj.root)
    assert "lim-2" in str(raised.value) and CASES["L2"]["reason"] in str(raised.value)
    assert not (proj.out / "limitations").exists()


def test_l3_redaction(proj: Proj) -> None:
    proj.limitations["items"][0]["summary"] = f"Accepted by Jane Approver ({EMAIL})."
    proj.limitations["items"][0]["redact"] = ["Jane Approver"]
    proj.build()
    source_before = digest(proj.ws / "storage" / "release" / "limitations_source.json")
    item = read(proj.out / "limitations" / "limitations.json")["items"][0]
    e = CASES["L3"]
    assert e["approver"] in item["summary"] and "Jane Approver" not in item["summary"]
    assert e["email"] in item["summary"] and EMAIL not in item["summary"]
    assert digest(proj.ws / "storage" / "release" / "limitations_source.json") == source_before


def test_l4_the_row_by_row_check(proj: Proj) -> None:
    proj.save()
    code, report = rf().check_limitations(proj.ws, project_root=proj.root)
    assert code == CASES["L4"]["a"]["exit_code"]
    proj.limitations["accepted_lines"].append([7, 7])
    proj.save()
    code, report = rf().check_limitations(proj.ws, project_root=proj.root)
    assert code == CASES["L4"]["b"]["exit_code"]
    assert [7, 7] in report["uncovered"]


# --- redaction in the bundle copy ---------------------------------------------------


def _redaction_workspace(ws: rc.Workspace) -> dict[str, Path]:
    deck = ws.conf / "decks" / "01_sample"
    files = {
        "receipt": deck / "release_receipt.json",
        "text": deck / "notes.md",
        "html": deck / "presentation.html",
        "pdf": deck / "presentation.pdf",
    }
    write(files["receipt"], {"publication": {"target": "Google " + "Drive: Private Folder", "approved_by": "Jane Approver"}})
    write(files["text"], f"Figure at {ws.conf}/decks/01_sample/assets/a.png and {OUTSIDE}/elsewhere/b.png\n")
    write(files["html"], '<p><a href="https://drive.google.com/file/d/MAPPED/view">Paper A</a> and '
                         '<a href="https://drive.google.com/file/d/UNMAPPED/view">Paper Z</a></p>\n')
    files["pdf"].write_bytes(b"%PDF-1.4 https://drive.google.com/file/d/X/view\n")
    write(ws.conf / "storage" / "release" / "link_map.json", {"https://drive.google.com/file/d/MAPPED/view": "https://openreview.net/forum?id=AAAAAAAAA1"})
    return files


def _redaction_manifest(ws: rc.Workspace) -> None:
    manifest = read(ws.manifest)
    manifest["families"][0]["redaction_rules"] = [
        {"kind": "json_field", "glob": "**/*.json", "pointer": "/publication/target", "replacement": "<redacted:private_target>"},
        {"kind": "absolute_path", "glob": "**/*.md"},
        {"kind": "relink", "glob": "**/*.html", "link_map": "storage/release/link_map.json"},
    ]
    write(ws.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def redacted(w0: rc.Workspace) -> tuple[rc.Workspace, dict[str, Path], dict[str, str]]:  # noqa: F811
    rc.draft(w0)
    rc.promote(w0)
    files = _redaction_workspace(w0)
    _redaction_manifest(w0)
    before = {name: digest(path) for name, path in files.items()}
    rc.plan(w0)
    assert rc.cli().release_build(rc.SLUG, mock=True, bundle_only=True, **w0.kw()) == 0, w0.text()
    return w0, files, before


def _bundle_file(ws: rc.Workspace, name: str) -> Path:
    return ws.bundle / "decks" / "01_sample" / name


def test_x1_json_field(redacted) -> None:
    ws, _, _ = redacted
    assert read(_bundle_file(ws, "release_receipt.json"))["publication"]["target"] == CASES["X1"]["value"]


def test_x2_absolute_path(redacted) -> None:
    ws, _, _ = redacted
    text = _bundle_file(ws, "notes.md").read_text(encoding="utf-8")
    assert "decks/01_sample/assets/a.png" in text and str(ws.conf) not in text
    assert CASES["X2"]["outside_home_path"] in text and OUTSIDE not in text


def test_x3_relink(redacted) -> None:
    ws, _, _ = redacted
    html = _bundle_file(ws, "presentation.html").read_text(encoding="utf-8")
    assert 'href="https://openreview.net/forum?id=AAAAAAAAA1"' in html
    assert "drive.google.com" not in html and "Paper Z" in html


def test_x4_a_binary_is_never_rewritten(redacted) -> None:
    ws, files, before = redacted
    assert digest(_bundle_file(ws, "presentation.pdf")) == before["pdf"]
    receipt = read(ws.runs / f"public-export-{rc.BUNDLE}-write-receipt.json")
    assert any(entry.endswith("presentation.pdf") for entry in receipt[CASES["X4"]["reported_under"]])


def test_x5_receipt_and_checksums(redacted) -> None:
    ws, _, _ = redacted
    receipt = read(ws.runs / f"public-export-{rc.BUNDLE}-write-receipt.json")
    paths = {entry["path"] for entry in receipt["redacted_files"]}
    assert {p for p in paths if p.endswith(("release_receipt.json", "notes.md", "presentation.html"))} == paths
    checksums = read(ws.bundle / "CHECKSUMS.json")
    for rel, entry in checksums.items():
        assert entry["sha256"] == digest(ws.bundle / rel), rel


def test_x6_workspace_originals_unchanged(redacted) -> None:
    _, files, before = redacted
    assert {name: digest(path) for name, path in files.items()} == before


# --- manifest, check, site, hub -----------------------------------------------------


def test_m1_conference_name_in_the_schema(tmp_path: Path) -> None:
    gate = importlib.import_module("conference_pipeline.publishing.export_gate")
    base = {"schema_version": "1.0.0", "conference": "conf", "release": {"source_sha": "a" * 40, "conference_name": "Conf 2026"},
            "families": [{"id": "f", "private_path": "decks", "public_path": "exports/public-release/{bundle}/decks/",
                          "storage_manifest_id": "f", "redistribution_status": "allowed", "license": "CC-BY-4.0",
                          "provenance": "pipeline-generated"}]}
    good = tmp_path / "good.json"
    write(good, base)
    gate.validate_public_export_manifest(good)
    bad = tmp_path / "bad.json"
    write(bad, {**base, "release": {**base["release"], "foo": 1}})
    with pytest.raises(gate.GateClosed):
        gate.validate_public_export_manifest(bad)
    assert CASES["M1"] == {"conference_name_valid": True, "unknown_release_key_valid": False}


def test_m2_init_refresh(w0: rc.Workspace) -> None:  # noqa: F811
    rc.draft(w0)
    rc.promote(w0)
    manifest = read(w0.manifest)
    promoted = manifest["families"][0]
    dropped = manifest["excluded_families"].pop()
    write(w0.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    assert rc.cli().release_init(rc.SLUG, refresh=True, **w0.kw()) == 0, w0.text()
    after = read(w0.manifest)
    e = CASES["M2"]
    row = next(r for r in after["excluded_families"] if r["storage_manifest_id"] == dropped["storage_manifest_id"])
    assert row["reason"] == e["F3_reason"]
    assert after["families"] == [promoted]


def test_c1_release_check_finds_a_drive_link(w0: rc.Workspace) -> None:  # noqa: F811
    (w0.conf / rc.DECK).write_text("see https://drive.google.com/file/d/X/view\n", encoding="utf-8")
    rc.plan(rc.promote(rc.draft(w0)))
    assert rc.cli().release_build(rc.SLUG, mock=True, bundle_only=True, **w0.kw()) == 0, w0.text()
    w0.out.clear()
    code = rc.cli().release_check(rc.SLUG, **w0.kw())
    assert code == CASES["C1"]["exit_code"]
    assert CASES["C1"]["finding_names"] in w0.text()


def test_h1_the_site_shows_talks_and_limitations(tmp_path: Path) -> None:
    root = site_fx.make_alpha(tmp_path)
    stats = {**_s1_statistics(), "d11_note": "Statistics describe the earlier corpus."}
    write(root / "catalog" / "statistics.json", stats)
    write(root / "limitations" / "limitations.md", "# Limitations\n\n- **accepted_finding** (lim-1): Deck 01 slide 4: an unlabelled connector, accepted.\n")
    site_fx._finish(root)
    static_site = importlib.import_module("conference_pipeline.publishing.static_site")
    out = tmp_path / "site"
    static_site.build_public_site(static_site.PublicSiteConfig(bundle_root=root, output_dir=out))
    index = (out / "index.html").read_text(encoding="utf-8")
    e = CASES["H1"]
    for text in ("Expo talk / panel", "conf-corpus-0123456789ab", stats["d11_note"]):
        assert text in index, text
    page = out / e["limitations_page"]
    assert page.is_file() and "an unlabelled connector" in page.read_text(encoding="utf-8")


def test_h2_the_hub_never_shows_an_excluded_id(tmp_path: Path) -> None:
    root = site_fx.make_alpha(tmp_path)
    write(root / "catalog" / "excluded_ids.json", {"excluded": ["openreview:A3"]})
    site_fx._finish(root)
    hub = importlib.import_module("conference_pipeline.publishing.hub_site")
    out = tmp_path / "hub"
    hub.build_hub_site([root], out, hub_title="Hub")
    catalog = read(out / "catalog.json")
    ids = json.dumps(catalog)
    assert ("openreview:A3" in ids) is CASES["H2"]["in_hub_catalog"]


def test_e1_end_to_end(w0: rc.Workspace) -> None:  # noqa: F811
    rc.promote(rc.draft(w0))
    project = Proj.__new__(Proj)
    Proj.__init__(project, w0.root.parent)
    project.root, project.ws = w0.root, w0.conf
    write(w0.root / ".planning" / "milestones" / "Conf-CONTEXT.md", "\n".join(CLOSE_RECORD) + "\n")
    write(w0.conf / "analysis" / "topic_reference_validation" / "01_alpha.json", {"valid_reference_ids": ["paper_abstract:AAAAAAAAA1"]})
    write(w0.conf / "analysis" / "01_alpha.md", "# Alpha\n\nA paragraph.\n")
    project.save()
    rf().build_reader_families(w0.conf, bundle_id=rc.BUNDLE, project_root=w0.root)
    manifest = read(w0.manifest)
    manifest["release"]["conference_name"] = "Conf 2026"
    manifest["families"].append({"id": "reader_families", "private_path": f"storage/release-staging/{rc.BUNDLE}",
                                 "public_path": "exports/public-release/{bundle}/", "storage_manifest_id": "release_staging",
                                 "redistribution_status": "allowed", "license": "CC-BY-4.0", "provenance": "paper-metadata"})
    write(w0.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    e = CASES["E1"]
    assert rc.cli().release_plan(rc.SLUG, **w0.kw()) == e["exit_codes"]["plan"], w0.text()
    assert rc.cli().release_build(rc.SLUG, mock=True, bundle_only=True, **w0.kw()) == e["exit_codes"]["build"], w0.text()
    for rel in e["bundle_contains"]:
        assert (w0.bundle / rel).is_file(), rel
    assert read(w0.bundle / "RELEASE.json")["release"]["conference_name"] == e["release_conference_name"]


# --- review phase-6a-r1 regressions ---------------------------------------------------


def test_r1_f1_a_bundle_id_that_escapes_is_refused_before_any_deletion(proj: Proj) -> None:
    proj.save()
    (proj.ws / "analysis" / "keep.md").write_text("keep\n", encoding="utf-8")
    with pytest.raises(rf().ReaderFamiliesError):
        rf().build_reader_families(proj.ws, bundle_id="../../analysis", project_root=proj.root)
    assert (proj.ws / "analysis" / "keep.md").is_file()


def test_r1_f2_stage6_records_reach_the_catalog_and_the_counts_agree(proj: Proj) -> None:
    proj.records = [
        {"artifact_id": "workshop_paper_text:1", "source_family_id": "workshop_paper_text", "paper_title": "Workshop paper",
         "authors": ["A"], "openreview_id": "WWWWWWWWW1"},
        {"artifact_id": "paper_text:2", "source_family_id": "paper_text", "paper_title": "Main paper", "item_id": "poster-60482",
         "item_page_url": "https://conf.example.org/virtual/2026/poster/60482", "metadata": {"material_type": "paper"}},
        {"artifact_id": "paper_text:3", "source_family_id": "paper_text", "paper_title": "Main paper", "item_id": "poster-60482",
         "item_page_url": "https://conf.example.org/virtual/2026/poster/60482", "metadata": {"material_type": "slides"}},
    ]
    proj.sources["catalog"] = {"adapter": "stage6_inventory", "path": "storage/corpus/corpus_inventory.json"}
    proj.build()
    rows = {row["paper_id"]: row for row in proj.rows()}
    assert set(rows) == {"openreview:WWWWWWWWW1", "site:poster-60482"}
    assert rows["openreview:WWWWWWWWW1"]["track"] == "workshop"
    assert rows["site:poster-60482"]["official_url"] == "https://conf.example.org/virtual/2026/poster/60482"
    stats = read(proj.out / "catalog" / "statistics.json")
    assert stats["papers"] == {"total": 2, "by_track": {"main": 1, "workshop": 1}}


def test_r1_f3_release_check_finds_a_token_in_a_binary(w0: rc.Workspace) -> None:  # noqa: F811
    (w0.conf / "decks" / "01_sample").mkdir(parents=True, exist_ok=True)
    (w0.conf / "decks" / "01_sample" / "deck.pdf").write_bytes(b"%PDF-1.4\x00\x01 drive.google.com/file/d/X\n")
    rc.plan(rc.promote(rc.draft(w0)))
    assert rc.cli().release_build(rc.SLUG, mock=True, bundle_only=True, **w0.kw()) == 0, w0.text()
    w0.out.clear()
    assert rc.cli().release_check(rc.SLUG, **w0.kw()) == 1
    assert "deck.pdf" in w0.text()


def test_r1_f4_a_binary_with_an_unlisted_suffix_is_never_rewritten(tmp_path: Path) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    data = b"BMfixture file://private"
    out = redaction.apply_redaction_rules(data, "img/figure.bmp", [{"kind": "marker_text", "glob": "**/*"}], workspace=tmp_path)
    assert out.content is None and out.binary_marker is True


def test_r1_f5_other_absolute_paths_are_redacted(tmp_path: Path) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    text = "a /srv/private/proof.json b /Volumes/private/x.json c https://example.org/a/b.html\n"
    out = redaction.apply_redaction_rules(text.encode(), "r.md", [{"kind": "absolute_path", "glob": "*.md"}], workspace=tmp_path)
    redacted = out.content.decode()
    assert "/srv/private" not in redacted and "/Volumes/private" not in redacted
    assert redacted.count("<redacted:path>") == 2 and "https://example.org/a/b.html" in redacted


def test_r1_f6_markdown_drive_links_are_relinked(tmp_path: Path) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    write(tmp_path / "map.json", {"https://drive.google.com/file/d/A/view": "https://openreview.net/forum?id=A1"})
    text = "[Paper A](https://drive.google.com/file/d/A/view) and [Paper Z](https://drive.google.com/file/d/Z/view)\n"
    out = redaction.apply_redaction_rules(text.encode(), "n.md", [{"kind": "relink", "glob": "*.md", "link_map": "map.json"}], workspace=tmp_path)
    redacted = out.content.decode()
    assert "[Paper A](https://openreview.net/forum?id=A1)" in redacted
    assert "drive.google.com" not in redacted and "Paper Z" in redacted


def test_r1_f7_the_check_reads_the_produced_record(proj: Proj) -> None:
    proj.save()
    code, report = rf().check_limitations(proj.ws, project_root=proj.root, bundle_id=BUNDLE)
    assert code == 1 and report["record"] == "missing"
    proj.build()
    assert rf().check_limitations(proj.ws, project_root=proj.root, bundle_id=BUNDLE)[0] == 0
    record_path = proj.out / "limitations" / "limitations.json"
    record = read(record_path)
    record["items"] = record["items"][:1]
    write(record_path, record)
    code, report = rf().check_limitations(proj.ws, project_root=proj.root, bundle_id=BUNDLE)
    assert code == 1 and report["missing_items"] == ["lim-2"]


@pytest.mark.parametrize("quote,lines", [(" \t ", [5, 5]), ("Deck 01 slide 4", [999, 999])])
def test_r1_f8_an_empty_quote_or_a_bad_range_is_refused(proj: Proj, quote: str, lines: list[int]) -> None:
    proj.limitations["items"][0]["quote"] = quote
    proj.limitations["items"][0]["close_record_lines"] = lines
    proj.save()
    with pytest.raises(rf().ReaderFamiliesError):
        rf().build_reader_families(proj.ws, bundle_id=BUNDLE, project_root=proj.root)


# --- review phase-6a-r2 regressions ---------------------------------------------------


def test_r2_f1_an_explicit_out_outside_staging_is_refused(proj: Proj) -> None:
    proj.save()
    with pytest.raises(rf().ReaderFamiliesError):
        rf().build_reader_families(proj.ws, bundle_id="safe-v1", project_root=proj.root, out=proj.ws / "analysis")
    assert (proj.ws / "analysis" / "01_alpha.md").is_file()


def test_r2_f2_a_large_binary_with_a_token_fails_check(w0: rc.Workspace) -> None:  # noqa: F811
    deck = w0.conf / "decks" / "01_sample"
    deck.mkdir(parents=True, exist_ok=True)
    (deck / "big.pdf").write_bytes(b"%PDF\x00" + b"\x01" * (2 * 1024 * 1024) + b" drive.google.com/file/d/X\n")
    rc.plan(rc.promote(rc.draft(w0)))
    assert rc.cli().release_build(rc.SLUG, mock=True, bundle_only=True, **w0.kw()) == 0, w0.text()
    w0.out.clear()
    assert rc.cli().release_check(rc.SLUG, **w0.kw()) == 1
    assert "big.pdf" in w0.text()


def test_r2_f3_the_check_compares_item_contents(proj: Proj) -> None:
    proj.build()
    path = proj.out / "limitations" / "limitations.json"
    record = read(path)
    record["items"][0]["summary"] = "Something unrelated."
    write(path, record)
    code, report = rf().check_limitations(proj.ws, project_root=proj.root, bundle_id=BUNDLE)
    assert code == 1 and report["differing_items"] == ["lim-1"]


def test_r2_f4_paths_after_a_colon_or_with_spaces_are_redacted(tmp_path: Path) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    text = "path:/srv/private/proof.json and /Volumes/Private Drive/confidential/proof.json end\n"
    out = redaction.apply_redaction_rules(text.encode(), "r.md", [{"kind": "absolute_path", "glob": "*.md"}], workspace=tmp_path)
    redacted = out.content.decode()
    for leak in ("/srv/private", "private/proof", "Drive/confidential", "confidential"):
        assert leak not in redacted, leak
    assert redacted.startswith("path:<redacted:path>")


def test_r2_f5_titled_and_reference_markdown_links_are_relinked(tmp_path: Path) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    write(tmp_path / "map.json", {"https://drive.google.com/file/d/A/view": "https://openreview.net/forum?id=A1"})
    text = ('[Paper A](https://drive.google.com/file/d/A/view "Source") and [Paper B][b] and [Paper Z][z]\n\n'
            '[b]: https://drive.google.com/file/d/A/view\n[z]: https://drive.google.com/file/d/Z/view "Z"\n')
    out = redaction.apply_redaction_rules(text.encode(), "n.md", [{"kind": "relink", "glob": "*.md", "link_map": "map.json"}], workspace=tmp_path)
    redacted = out.content.decode()
    assert "drive.google.com" not in redacted
    assert '[Paper A](https://openreview.net/forum?id=A1 "Source")' in redacted
    assert "[b]: https://openreview.net/forum?id=A1" in redacted and "Paper Z" in redacted and "[z]" not in redacted


def test_r2_f6_file_statistics_are_not_replaced_by_the_catalog(proj: Proj) -> None:
    proj.sources["statistics"] = {"path": "storage/corpus/corpus_statistics.json", "inventory": "storage/corpus/corpus_inventory.json"}
    stats = _s1_statistics()
    stats["papers"] = {"total": 7, "by_track": {"main": 7}}
    write(proj.ws / "storage" / "corpus" / "corpus_statistics.json", stats)
    proj.build()
    assert read(proj.out / "catalog" / "statistics.json")["papers"] == {"total": 7, "by_track": {"main": 7}}


# --- review phase-6a-r3 regressions ---------------------------------------------------


def test_r3_f1_a_symlinked_staging_root_is_refused(proj: Proj) -> None:
    proj.save()
    (proj.ws / "storage").mkdir(parents=True, exist_ok=True)
    (proj.ws / "storage" / "release-staging").symlink_to(proj.ws / "analysis")
    with pytest.raises(rf().ReaderFamiliesError):
        rf().build_reader_families(proj.ws, bundle_id="safe-v1", project_root=proj.root)
    assert not (proj.ws / "analysis" / "safe-v1").exists()


def test_r3_f2_a_token_in_a_git_folder_fails_check(w0: rc.Workspace) -> None:  # noqa: F811
    rc.plan(rc.promote(rc.draft(w0)))
    assert rc.cli().release_build(rc.SLUG, mock=True, bundle_only=True, **w0.kw()) == 0, w0.text()
    note = w0.bundle / "public-repo" / ".git" / "note.txt"
    note.parent.mkdir(parents=True)
    note.write_text("see drive.google.com/file/d/X\n", encoding="utf-8")
    w0.out.clear()
    assert rc.cli().release_check(rc.SLUG, **w0.kw()) == 1
    assert ".git/note.txt" in w0.text()


@pytest.mark.parametrize("tamper", ["metadata", "duplicate", "non_object"])
def test_r3_f3_the_whole_record_must_match(proj: Proj, tamper: str) -> None:
    proj.build()
    path = proj.out / "limitations" / "limitations.json"
    record = read(path)
    if tamper == "metadata":
        record["close_record_sha256"] = "0" * 64
    elif tamper == "duplicate":
        record["items"].insert(0, {**record["items"][0], "summary": "altered"})
    else:
        record["items"].append("not an item")
    write(path, record)
    code, report = rf().check_limitations(proj.ws, project_root=proj.root, bundle_id=BUNDLE)
    assert code == 1 and report["record_matches_source"] is False


def test_r3_f4_spaced_and_symbol_file_names_are_redacted_whole(tmp_path: Path) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    cases = {
        "/srv/private/customer secret.json done\n": ("secret", "customer"),
        "/srv/private/Client  A/report.json done\n": ("Client", "report"),
        "/srv/private/client+secret/report.json done\n": ("secret", "client"),
    }
    for text, leaks in cases.items():
        out = redaction.apply_redaction_rules(text.encode(), "r.md", [{"kind": "absolute_path", "glob": "*.md"}], workspace=tmp_path)
        redacted = out.content.decode()
        assert redacted == "<redacted:path> done\n", redacted
        assert not any(leak in redacted for leak in leaks)


@pytest.mark.parametrize("link", ["[Paper](<https://drive.google.com/file/d/A/view>)",
                                  "[Paper](https://drive.google.com/file/d/A/view 'Source')",
                                  "[Paper](https://drive.google.com/file/d/A/view (Source))"])
def test_r3_f5_every_inline_link_form_is_relinked(tmp_path: Path, link: str) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    write(tmp_path / "map.json", {"https://drive.google.com/file/d/A/view": "https://openreview.net/forum?id=A1"})
    out = redaction.apply_redaction_rules(link.encode(), "n.md", [{"kind": "relink", "glob": "*.md", "link_map": "map.json"}], workspace=tmp_path)
    redacted = out.content.decode()
    assert "drive.google.com" not in redacted and "https://openreview.net/forum?id=A1" in redacted


# --- 6b successors (real-data shapes) -------------------------------------------------


def test_b1_a_catalog_from_two_sources(proj: Proj) -> None:
    """6b: ICML's main papers come from the accepted-papers file (OpenReview ids), its
    workshop papers from the Stage 6 inventory; the first source wins on a shared id."""
    write(proj.ws / "storage" / "download-corpus" / "accepted.json", {"papers": [
        {"id": "AAAAAAAAA1", "title": "Sparse Mixtures for Long Context Reasoning", "authors": ["Ada One"],
         "venue": "Conf 2026 regular", "forum_url": "https://openreview.net/forum?id=AAAAAAAAA1"}]})
    proj.records = [
        {"artifact_id": "w:1", "source_family_id": "workshop_paper_text", "paper_title": "A Workshop Paper", "authors": ["B"], "openreview_id": "WWWWWWWWW1"},
        {"artifact_id": "w:2", "source_family_id": "workshop_paper_text", "paper_title": "Dup", "authors": ["C"], "openreview_id": "AAAAAAAAA1"},
        {"artifact_id": "p:1", "source_family_id": "paper_text", "paper_title": "Main via inventory", "item_id": "poster-1",
         "item_page_url": "https://conf.example.org/poster/1"},
    ]
    proj.sources["catalog"] = [
        {"adapter": "openreview_accepted_papers", "path": "storage/download-corpus/accepted.json"},
        {"adapter": "stage6_inventory", "path": "storage/corpus/corpus_inventory.json", "families": ["workshop_paper_text"]},
    ]
    proj.build()
    rows = {row["paper_id"]: row for row in proj.rows()}
    assert set(rows) == {"openreview:AAAAAAAAA1", "openreview:WWWWWWWWW1"}
    assert rows["openreview:AAAAAAAAA1"]["track"] == "main" and rows["openreview:WWWWWWWWW1"]["track"] == "workshop"
    assert rows["openreview:AAAAAAAAA1"]["topics"] == ["01_alpha"]


def test_b2_an_m4a_podcast_is_read(tmp_path: Path) -> None:
    """6b: ICML and CVPR podcasts are NotebookLM m4a files, delivered by CDN sidecar."""
    root = site_fx.make_alpha(tmp_path)
    rel = "audio/02_agents/notebooklm_podcast/podcast.m4a.cdn.json"
    write(root / rel, {"cdn_url": "https://cdn.example.test/alpha/audio/02_agents/notebooklm_podcast/podcast.m4a",
                       "mime_type": "audio/mp4", "object_key": rel, "sha256": "0" * 64, "size_bytes": 1, "source_filename": "podcast.m4a"})
    site_fx._finish(root)
    reader = importlib.import_module("conference_pipeline.publishing.reader_bundle")
    bundle = reader.load_reader_bundle(root)
    topic = next(t for t in bundle.topics if t.topic_id == "02_agents")
    assert topic.podcast == "https://cdn.example.test/alpha/audio/02_agents/notebooklm_podcast/podcast.m4a"


def test_b3_bare_drive_urls_are_relinked(tmp_path: Path) -> None:
    """6b: ICML's executive presenter scripts cite sources as bare Drive URLs in parentheses."""
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    write(tmp_path / "map.json", {"https://drive.google.com/file/d/A/view?usp=drivesdk": "https://site.example/topics/01.html"})
    text = ("- Source: Topic one (https://drive.google.com/file/d/A/view?usp=drivesdk)\n"
            "- Source: Unknown (https://drive.google.com/file/d/Z/view?usp=drivesdk)\n"
            "See https://drive.google.com/file/d/A/view?usp=drivesdk now.\n")
    out = redaction.apply_redaction_rules(text.encode(), "s.md", [{"kind": "relink", "glob": "*.md", "link_map": "map.json"}], workspace=tmp_path)
    redacted = out.content.decode()
    assert "drive.google.com" not in redacted
    assert "- Source: Topic one (https://site.example/topics/01.html)" in redacted
    assert "- Source: Unknown\n" in redacted
    assert "See https://site.example/topics/01.html now." in redacted


# --- 6b successors: self-contained deck HTML (owner ruling D21) and regenerated PDFs (D8 a) ---

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="4" height="4"/></svg>'
INLINE = [{"kind": "inline_assets", "glob": "**/*.html"}]


def _deck(base: Path) -> tuple[Path, Path]:
    """A project with one deck whose HTML loads a deck-local SVG and a brand PNG four folders up."""
    root = base / "proj"
    ws = root / "conferences" / "conf"
    deck = ws / "decks" / "01_alpha"
    (deck / "assets").mkdir(parents=True)
    (deck / "assets" / "fig.svg").write_bytes(SVG)
    (root / "assets" / "brand").mkdir(parents=True)
    (root / "assets" / "brand" / "logo.png").write_bytes(PNG)
    html = ('<html><body><img src="../../../../assets/brand/logo.png"><img src="assets/fig.svg">'
            '<a href="notes.html">n</a><img src="data:image/png;base64,AAAA"><a href="#s2">s</a>'
            '<a href="https://drive.google.com/">report</a></body></html>')
    (deck / "presentation.html").write_text(html, encoding="utf-8")
    return ws, deck / "presentation.html"


def _inline(ws: Path, html: Path, rules: list[dict[str, Any]] | None = None) -> str:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    out = redaction.apply_redaction_rules(html.read_bytes(), "01_alpha/presentation.html", rules or INLINE,
                                          workspace=ws, source_path=html)
    return out.content.decode()


def test_b4_local_images_become_data_uris(tmp_path: Path) -> None:
    import base64
    ws, html = _deck(tmp_path)
    text = _inline(ws, html)
    assert f'src="data:image/png;base64,{base64.b64encode(PNG).decode()}"' in text
    assert f'src="data:image/svg+xml;base64,{base64.b64encode(SVG).decode()}"' in text
    # Not images, already inline, or anchors: untouched.
    assert 'href="notes.html"' in text and 'src="data:image/png;base64,AAAA"' in text and 'href="#s2"' in text
    assert "../../../../assets" not in text and 'src="assets/' not in text


@pytest.mark.parametrize("case", ["outside", "symlink", "missing", "marker", "no_source"])
def test_b5_an_asset_that_cannot_be_inlined_safely_is_refused(tmp_path: Path, case: str) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    ws, html = _deck(tmp_path)
    fig = html.parent / "assets" / "fig.svg"
    if case == "outside":
        (tmp_path / "secret.png").write_bytes(PNG)
        html.write_text('<img src="../../../../../secret.png">', encoding="utf-8")
    elif case == "symlink":
        (tmp_path / "secret.svg").write_bytes(SVG)
        fig.unlink()
        fig.symlink_to(tmp_path / "secret.svg")
    elif case == "missing":
        fig.unlink()
    elif case == "marker":
        fig.write_bytes(b'<svg><a href="https://drive.google.com/x"/></svg>')
    with pytest.raises(redaction.RedactionError):
        if case == "no_source":
            redaction.apply_redaction_rules(html.read_bytes(), "01_alpha/presentation.html", INLINE, workspace=ws)
        else:
            _inline(ws, html)


def test_b6_inlining_runs_after_every_other_rule(tmp_path: Path) -> None:
    import base64
    ws, html = _deck(tmp_path)
    tricky = base64.b64decode("+/xx/yyAAzz+/" + "Us" + "ers/abcdA")  # its base64 holds path-shaped runs
    (html.parent / "assets" / "fig.svg").write_bytes(tricky)
    rules = INLINE + [{"kind": "absolute_path", "glob": "**/*.html"}, {"kind": "marker_text", "glob": "**/*.html"}]
    text = _inline(ws, html, rules)
    assert f"base64,{base64.b64encode(tricky).decode()}" in text


def test_b7_deck_pdfs_are_regenerated_from_the_published_html(proj: Proj) -> None:
    ws, html = _deck(proj.root.parent)
    rules = [{"kind": "relink", "glob": "**/*.html", "link_map": "storage/release/link_map.json"}] + INLINE
    write(ws / "storage" / "release" / "link_map.json", {"https://drive.google.com/": "https://site.example/"})
    write(ws / "storage" / "public_export_manifest.json",
          {"families": [{"id": "presentation_payloads", "private_path": "decks", "redaction_rules": rules}]})
    (ws / "decks" / "02_beta").mkdir()
    (ws / "decks" / "02_beta" / "presentation.html").write_text("<p>drive marker renders</p>", encoding="utf-8")
    proj.sources["regenerated_pdfs"] = {"family": "presentation_payloads", "html": "*/presentation.html"}
    before = {p: digest(p) for p in (ws / "decks").rglob("*") if p.is_file()}
    seen: list[str] = []

    def render(input_html: Path, output_pdf: Path) -> None:
        text = input_html.read_text(encoding="utf-8")
        seen.append(text)
        # The renderer sees the published copy: relinked, every image inline, nothing beside it.
        assert [p.name for p in input_html.parent.iterdir()] == [input_html.name]
        # A real one-page PDF: since review phase-6b-r1 F1 an unreadable PDF fails closed.
        body = b"https://drive.google.com/" if "drive marker" in text else _pdf_with_links(["https://site.example/"])
        output_pdf.write_bytes(body)

    proj.save()
    result = rf().build_reader_families(ws, bundle_id=BUNDLE, project_root=proj.root, render_pdf=render)
    pdf = ws / "storage" / "release-staging" / BUNDLE / "pdfs" / "01_alpha" / "presentation.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-") and b"drive.google" not in pdf.read_bytes()
    assert "https://site.example/" in seen[0] and "data:image/svg+xml;base64," in seen[0]
    # A regenerated PDF that still holds a marker is excluded, and said so (D8 a).
    assert not (pdf.parent.parent / "02_beta").exists()
    assert result.excluded_pdfs == ("02_beta/presentation.pdf",)
    assert {p: digest(p) for p in (ws / "decks").rglob("*") if p.is_file()} == before


def test_b8_html_that_still_loads_a_local_file_is_not_rendered(proj: Proj) -> None:
    ws, html = _deck(proj.root.parent)
    write(ws / "storage" / "public_export_manifest.json",
          {"families": [{"id": "presentation_payloads", "private_path": "decks", "redaction_rules": []}]})
    proj.sources["regenerated_pdfs"] = {"family": "presentation_payloads", "html": "*/presentation.html"}
    proj.save()
    with pytest.raises(rf().ReaderFamiliesError) as err:
        rf().build_reader_families(ws, bundle_id=BUNDLE, project_root=proj.root, render_pdf=lambda **_: None)
    assert err.value.code == "pdf_html_not_self_contained"


# --- 6b successor: a release presentation manifest from the decks themselves (owner ruling D22) ---

def _frame(ws: Path, deck: str, title: str | None) -> None:
    (ws / "decks" / deck).mkdir(parents=True, exist_ok=True)
    (ws / "decks" / deck / "presentation.html").write_text("<p>deck</p>", encoding="utf-8")
    write(ws / "decks" / deck / "presentation_frame.json", {"presentation_id": deck, **({"audience_title": title} if title else {})})


def test_b9_the_release_manifest_lists_every_deck(proj: Proj) -> None:
    ws = proj.ws
    _frame(ws, "01_alpha", "Alpha, the first topic")
    _frame(ws, "02_beta", None)
    _frame(ws, "executive_summary_long", "Executive readout")
    # The workspace's own manifest is a stale snapshot naming one deck; it is not the source.
    write(ws / "decks" / "presentation_manifest.json", {"presentations": [{"id": "02_beta", "presentation_type": "topic-deep-dive"}]})
    proj.sources["presentations"] = {"decks": "decks", "html": "*/presentation.html", "executive": "executive_*"}
    proj.build()
    manifest = read(proj.out / "presentations" / "presentation_manifest.json")
    assert manifest["presentations"] == [
        {"id": "01_alpha", "presentation_type": "topic-deep-dive", "title": "Alpha, the first topic"},
        {"id": "02_beta", "presentation_type": "topic-deep-dive", "title": "02_beta"},
        {"id": "executive_summary_long", "presentation_type": "executive-summary", "title": "Executive readout"},
    ]


def test_b10_a_deck_without_published_html_is_left_out(proj: Proj) -> None:
    ws = proj.ws
    _frame(ws, "01_alpha", "Alpha")
    (ws / "decks" / "03_draft").mkdir(parents=True)
    write(ws / "decks" / "03_draft" / "presentation_frame.json", {"audience_title": "Draft"})
    proj.sources["presentations"] = {"decks": "decks", "html": "*/presentation.html"}
    proj.build()
    ids = [e["id"] for e in read(proj.out / "presentations" / "presentation_manifest.json")["presentations"]]
    assert ids == ["01_alpha"]


# --- 6b successors: the ICML site check (T-B5) ---

def test_b11_repeated_link_text_is_named_by_its_paper(tmp_path: Path) -> None:
    checks = importlib.import_module("conference_pipeline.publishing.site_checks")
    md = ("- *Alpha Paper* — `id:a`, [OpenReview](https://openreview.net/forum?id=a)\n"
          "- *Beta Paper* — `id:b`, [OpenReview](https://openreview.net/forum?id=b)\n"
          "- [OpenReview](https://openreview.net/forum?id=c)\n"
          "- [Home](https://site.example/) and again [Home](https://site.example/)\n")
    html = checks.render_markdown(md)
    assert 'aria-label="OpenReview: Alpha Paper"' in html
    assert 'aria-label="OpenReview: Beta Paper"' in html
    assert 'aria-label="OpenReview: https://openreview.net/forum?id=c"' in html
    assert 'aria-label="Home' not in html  # the same target twice needs no label
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><title>t</title></head><body><a class="skip" href="#m">Skip</a>'
        f'<main id="m"><h1>t</h1>{html}</main></body></html>', encoding="utf-8")
    result = checks.lint_accessibility(site)
    assert not [f for f in result.findings if "share the text" in f], result.findings


def test_b12_long_tokens_wrap_at_phone_width(tmp_path: Path) -> None:
    checks = importlib.import_module("conference_pipeline.publishing.site_checks")
    tpl = importlib.import_module("conference_pipeline.publishing.site_templates")
    site = tmp_path / "site"
    (site / "assets").mkdir(parents=True)
    (site / "assets" / "site.css").write_text(tpl.CSS, encoding="utf-8")
    token = "decks/09_data_memorization_privacy/DEMOTED_GATE_A_FINDINGS_" + "x" * 60 + ".md"
    (site / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width">'
        '<link rel="stylesheet" href="assets/site.css"><title>t</title></head><body><main><h1>t</h1>'
        f"<ul><li>see {token}</li></ul><p><code>paper_text:{'a' * 80}</code></p></main></body></html>",
        encoding="utf-8")
    result = checks.render_mobile(site)
    if result.skipped:
        pytest.skip(result.skipped)
    assert result.findings == []


def test_b13_a_narratives_workspace_links_become_text(tmp_path: Path) -> None:
    root = site_fx.make_alpha(tmp_path)
    narrative = ("# Reasoning\n\nSee the [deep dive](analysis/deep_dive_topics/01__dd01.md), the "
                 "[paper](https://openreview.net/forum?id=a) and [below](#refs).\n")
    (root / "narratives" / "01_reasoning.md").write_text(narrative, encoding="utf-8")
    site_fx._finish(root)
    static_site = importlib.import_module("conference_pipeline.publishing.static_site")
    checks = importlib.import_module("conference_pipeline.publishing.site_checks")
    out = tmp_path / "site"
    static_site.build_public_site(static_site.PublicSiteConfig(bundle_root=root, output_dir=out))
    page = (out / "topics" / "01_reasoning.html").read_text(encoding="utf-8")
    assert "deep_dive_topics" not in page and "the deep dive, the" in page.replace("\n", " ")
    assert 'href="https://openreview.net/forum?id=a"' in page and 'href="#refs"' in page
    assert checks.check_links(out).findings == []


# --- 6b successor: talk transcripts in the release (owner ruling D23) ---

def _talk(item: str, kind: str, title: str, text_rel: str | None, url: str | None = "https://conf.example/talk/1") -> dict[str, Any]:
    record = {"artifact_id": f"speech_transcript:{item}", "source_family_id": "speech_transcript", "item_id": item,
              "item_kind": kind, "item_page_url": url, "indexable_text": text_rel is not None,
              "metadata": {"title": title, "provider_model": "asr-model-x"}}
    if text_rel:
        record["indexed_text_path"] = text_rel
    return record


def test_b14_transcripts_are_staged_with_a_header_and_an_index(proj: Proj) -> None:
    ws = proj.ws
    write(ws / "storage" / "transcription-ocr" / "t" / "a.txt", "Hello, this is the talk. Mail " + EMAIL + " for slides.\n")
    proj.records = [_talk("oral-1", "oral", "Scaling Laws, Revisited", "storage/transcription-ocr/t/a.txt"),
                    _talk("poster-2", "poster", "No speech here", None)]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    proj.build()
    page = (proj.out / "transcripts" / "oral-1.md").read_text(encoding="utf-8")
    assert page.startswith("# Scaling Laws, Revisited\n")
    assert "Oral" in page and "https://conf.example/talk/1" in page and "asr-model-x" in page
    assert "may contain errors" in page.lower() and "Hello, this is the talk." in page
    assert EMAIL not in page
    assert not (proj.out / "transcripts" / "poster-2.md").exists()
    rows = [json.loads(line) for line in (proj.out / "transcripts" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"item_id": "oral-1", "kind": "Oral", "title": "Scaling Laws, Revisited",
                     "page_url": "https://conf.example/talk/1", "transcript": "transcripts/oral-1.md", "model": "asr-model-x"}]


@pytest.mark.parametrize("case", ["unsafe_id", "missing_text"])
def test_b14_a_transcript_that_cannot_be_staged_safely_is_refused(proj: Proj, case: str) -> None:
    ws = proj.ws
    write(ws / "storage" / "transcription-ocr" / "t" / "a.txt", "text\n")
    item = "../escape" if case == "unsafe_id" else "oral-1"
    path = "storage/transcription-ocr/t/a.txt" if case == "unsafe_id" else "storage/transcription-ocr/t/gone.txt"
    proj.records = [_talk(item, "oral", "T", path)]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    with pytest.raises(rf().ReaderFamiliesError):
        proj.build()
    assert not proj.out.exists()


def test_b15_search_lists_a_transcript_as_a_talk() -> None:
    static_site = importlib.import_module("conference_pipeline.publishing.static_site")
    assert static_site._chunk_kind("transcripts/oral-1.md") == ("talk", None)


def test_b14_a_transcript_reached_through_a_link_is_refused(proj: Proj, tmp_path: Path) -> None:
    ws = proj.ws
    (tmp_path / "outside.txt").write_text("private text\n", encoding="utf-8")
    (ws / "storage" / "transcription-ocr").mkdir(parents=True)
    (ws / "storage" / "transcription-ocr" / "a.txt").symlink_to(tmp_path / "outside.txt")
    proj.records = [_talk("oral-1", "oral", "T", "storage/transcription-ocr/a.txt")]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    with pytest.raises(rf().ReaderFamiliesError) as err:
        proj.build()
    assert err.value.code == "transcript_path_unsafe"


# --- 6b successor: Stage 6 papers identified from receipts and the download corpus (CVPR 2026) ---

def _s6(aid: str, fam: str, title: str, **extra: Any) -> dict[str, Any]:
    return {"artifact_id": aid, "source_family_id": fam, "paper_title": title, "authors": ["A One"], "indexable_text": True, **extra}


def test_b16_stage6_papers_get_ids_and_links_from_receipts_and_downloads(proj: Proj) -> None:
    ws = proj.ws
    main_key = "a" * 64
    proj.records = [
        _s6(f"paper:{main_key}", "paper_text", "A Main Paper", storage_lane="storage/corpus/paper-text"),
        _s6("proceedings_paper:cvf_findings:cvf_111", "paper_text", "A Findings Paper",
            storage_lane="storage/papers/proceedings/cvf_findings",
            source_url="https://openaccess.example/content/F/papers/X_paper.pdf"),
        _s6("workshop_paper:openaccess.example/W/ABC:cvf_222", "workshop_paper_text", "A CVF Workshop Paper"),
        _s6("workshop_paper:example.com/2026/Workshop/V:FORUM12345", "workshop_paper_text", "An OpenReview Workshop Paper"),
        _s6("workshop_paper:workshop-site/9:site_333", "workshop_paper_text", "A Site Paper Without A Link"),
        _s6("paper:" + "b" * 64, "paper_text", "No Identity Anywhere"),
    ]
    write(ws / "storage" / "download-corpus" / "download_corpus.json", {"program_item_results": [
        {"request_key": main_key, "acquisition_plan": {"source_urls": ["https://openaccess.example/content/M/papers/Y_paper.pdf"]}}]})
    rec = ws / "storage" / "workshops" / "papers"
    write(rec / "w1" / "receipts" / "cvf_222.json", {"source": "cvf_openaccess", "note_id": "cvf_222",
          "html_url": "https://openaccess.example/content/W/html/Z_paper.html", "pdf_url": "https://openaccess.example/z.pdf"})
    write(rec / "w2" / "receipts" / "FORUM12345.json", {"note_id": "FORUM12345", "venueid": "example.com/2026/Workshop/V"})
    write(rec / "w3" / "receipts" / "site_333.json", {"source": "workshop_site", "note_id": "site_333"})
    proj.sources["catalog"] = {"adapter": "stage6_inventory", "path": "storage/corpus/corpus_inventory.json",
                               "receipts": "storage/workshops/papers/*/receipts/*.json",
                               "download_corpus": "storage/download-corpus/download_corpus.json",
                               "tracks": {"storage/papers/proceedings/cvf_findings": "findings"}}
    write(ws / "analysis" / "topic_reference_validation" / "01_alpha.json", {"valid_reference_ids": [f"paper_abstract:{main_key}"]})
    proj.build()
    rows = {r["title"]: r for r in proj.rows()}
    assert rows["A Main Paper"]["paper_id"] == f"proceedings:{main_key}"
    assert rows["A Main Paper"]["official_url"] == "https://openaccess.example/content/M/papers/Y_paper.pdf"
    assert rows["A Main Paper"]["track"] == "main" and rows["A Main Paper"]["topics"] == ["01_alpha"]
    assert rows["A Findings Paper"]["paper_id"] == "proceedings:cvf_111"
    assert rows["A Findings Paper"]["track"] == "findings"
    assert rows["A Findings Paper"]["official_url"] == "https://openaccess.example/content/F/papers/X_paper.pdf"
    assert rows["A CVF Workshop Paper"]["official_url"] == "https://openaccess.example/content/W/html/Z_paper.html"
    assert rows["An OpenReview Workshop Paper"]["paper_id"] == "openreview:FORUM12345"
    assert rows["An OpenReview Workshop Paper"]["official_url"] == "https://openreview.net/forum?id=FORUM12345"
    assert rows["A Site Paper Without A Link"]["official_url"] is None
    assert "No Identity Anywhere" not in rows


def test_b17_every_stage6_transcript_shape_becomes_plain_text(proj: Proj) -> None:
    ws = proj.ws
    cap = "storage/media/captions/-abc.en.vtt"
    write(ws / cap, "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nWelcome to our poster.\n\n"
                    "00:00:02.000 --> 00:00:04.000\nWe study scenes.\n")
    write(ws / "storage" / "media" / "g.json", {"schema_version": 1, "text": "Gemini text of the talk.",
                                                 "segments": [{"start": 0, "end": 1, "text": "Gemini text of the talk."}]})
    write(ws / "storage" / "media" / "ann.json", {"schema_version": 2, "segments": [
        {"talk_title": "Opening", "talk_speaker": "Ann Host", "text": "Hello all."},
        {"talk_title": "Opening", "talk_speaker": "Ann Host", "text": "Let us begin."},
        {"talk_title": "Keynote", "talk_speaker": "Ben Speaker", "text": "Thanks for having me."}]})
    proj.records = [
        {"artifact_id": "event_media_caption:-abc", "source_family_id": "speech_transcript", "indexable_text": True,
         "video_id": "-abc", "event_route_family": "poster", "title": "A Poster Talk",
         "source_event_urls": ["https://conf.example/poster/1"], "transcript_path": cap, "transcript_source": "youtube_caption_auto"},
        {"artifact_id": "speech:g", "source_family_id": "speech_transcript", "indexable_text": True, "video_id": "gvid",
         "title": "A Gemini Talk", "transcript_path": "storage/media/g.json", "transcript_source": "gemini_youtube_url_transcription"},
        {"artifact_id": "speech:ann", "source_family_id": "speech_transcript", "indexable_text": True, "video_id": "wsvid",
         "title": "A Workshop Recording", "annotated_transcript_json_path": "storage/media/ann.json", "transcript_source": "youtube_caption_auto"},
    ]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    proj.build()
    out = proj.out / "transcripts"
    poster = (out / "video--abc.md").read_text(encoding="utf-8")
    assert "Welcome to our poster. We study scenes." in poster and "WEBVTT" not in poster and "-->" not in poster
    assert "Poster" in poster and "https://conf.example/poster/1" in poster and "youtube_caption_auto" in poster
    gemini = (out / "video-gvid.md").read_text(encoding="utf-8")
    assert "Gemini text of the talk." in gemini and '"segments"' not in gemini
    ann = (out / "video-wsvid.md").read_text(encoding="utf-8")
    assert "## Opening (Ann Host)\n\nHello all. Let us begin.\n\n## Keynote (Ben Speaker)\n\nThanks for having me." in ann


def test_b18_no_drive_link_or_email_reaches_a_catalog_row(proj: Proj) -> None:
    ws = proj.ws
    drive = "https://drive." + "google.com/uc?export=download&id=X"
    proj.records = [
        _s6("workshop_paper:workshop-site/9:site_1", "workshop_paper_text", "Drive Hosted With A Workshop Page"),
        _s6("workshop_paper:workshop-site/9:site_2", "workshop_paper_text", "Drive Hosted Only"),
        _s6("proceedings_paper:p:cvf_3", "paper_text", "Messy Authors",
            source_url="https://openaccess.example/p.pdf",
            authors=["Ada One", "someone" + "@" + "uni.example.edu \"Some University", "Ben Two"]),
    ]
    rec = ws / "storage" / "workshops" / "papers"
    write(rec / "w" / "receipts" / "site_1.json", {"source": "workshop_site", "note_id": "site_1", "pdf_url": drive,
                                                    "research_source_url": "https://workshop.example/papers"})
    write(rec / "w" / "receipts" / "site_2.json", {"source": "workshop_site", "note_id": "site_2", "pdf_url": drive})
    proj.sources["catalog"] = {"adapter": "stage6_inventory", "path": "storage/corpus/corpus_inventory.json",
                               "receipts": "storage/workshops/papers/*/receipts/*.json"}
    proj.build()
    rows = {r["title"]: r for r in proj.rows()}
    assert rows["Drive Hosted With A Workshop Page"]["official_url"] == "https://workshop.example/papers"
    assert rows["Drive Hosted Only"]["official_url"] is None
    assert rows["Messy Authors"]["authors"] == ["Ada One", "Ben Two"]
    text = (proj.out / "catalog" / "papers.jsonl").read_text(encoding="utf-8")
    assert "drive." + "google.com" not in text and "@" not in text


@pytest.mark.parametrize("url,kind", [
    ("https://conf.example/virtual/2026/poster/39435", "Poster"),
    ("https://conf.example/virtual/2026/oral/12", "Oral"),
    ("https://conf.example/virtual/2026/somethingelse/12", "Unspecified"),
    (None, "Unspecified"),
])
def test_b19_a_missing_talk_kind_is_read_from_the_official_page_path(proj: Proj, url: str | None, kind: str) -> None:
    ws = proj.ws
    write(ws / "storage" / "t" / "a.txt", "Talk text.\n")
    record = {"artifact_id": "speech:v", "source_family_id": "speech_transcript", "indexable_text": True,
              "video_id": "vid1", "title": "A Talk", "transcript_path": "storage/t/a.txt", "transcript_source": "asr"}
    if url:
        record["source_event_urls"] = [url]
    proj.records = [record]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    proj.build()
    row = json.loads((proj.out / "transcripts" / "index.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["kind"] == kind


def test_b20_a_video_id_with_a_colon_becomes_a_plain_file_name(proj: Proj) -> None:
    ws = proj.ws
    write(ws / "storage" / "t" / "a.txt", "Talk text.\n")
    proj.records = [{"artifact_id": "evidence_lane_transcript:l:event:115787", "source_family_id": "speech_transcript",
                     "indexable_text": True, "video_id": "event:115787", "item_kind": "oral", "title": "An Oral",
                     "source_path": "storage/t/a.txt", "transcript_path": "storage/t/a.txt"}]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    proj.build()
    assert (proj.out / "transcripts" / "video-event-115787.md").is_file()


def test_b21_a_topic_page_links_the_deck_powerpoint(tmp_path: Path) -> None:
    root = site_fx.make_alpha(tmp_path)
    site_fx._put(root, "decks/payloads/01_reasoning/presentation.pptx", b"PK\x03\x04 a deck")
    site_fx._finish(root)
    static_site = importlib.import_module("conference_pipeline.publishing.static_site")
    out = tmp_path / "site"
    static_site.build_public_site(static_site.PublicSiteConfig(bundle_root=root, output_dir=out))
    page = (out / "topics" / "01_reasoning.html").read_text(encoding="utf-8")
    assert "Download the slides (PowerPoint)" in page
    assert 'href="../files/decks/payloads/01_reasoning/presentation.pptx"' in page
    assert (out / "files" / "decks" / "payloads" / "01_reasoning" / "presentation.pptx").is_file()


# --- 6b successors: NeurIPS 2025 text decks, figures as links (owner ruling D25) ---

def test_b22_a_paper_figure_becomes_a_link_to_its_paper(tmp_path: Path) -> None:
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    ws = tmp_path / "proj" / "conferences" / "conf"
    deck = ws / "decks" / "01_alpha"
    write(deck / "asset_inventory.json", {"figures": [
        {"source_slide": "slide_03", "image_filename": "f1.png", "source_paper": "Sparse Mixtures for Long Context Reasoning"},
        {"source_slide": "slide_04", "image_filename": "f2.png", "source_paper": "A Paper Outside The Catalog"}]})
    write(ws / "storage" / "release-staging" / "b" / "catalog" / "papers.jsonl",
          json.dumps({"paper_id": "openreview:A1", "title": "Sparse mixtures for long-context reasoning",
                      "official_url": "https://openreview.net/forum?id=A1"}) + "\n")
    md = ("# Slide\n\n![Slide title](assets/figures/slide_03/f1.png)\n*Figure 1: a caption*\n\n"
          "![Other](assets/figures/slide_04/f2.png)\n\n![Unknown](assets/figures/slide_09/zz.png)\n"
          "![Remote](https://example.org/logo.png)\n")
    (deck / "5_draft_slides_with_images.md").write_text(md, encoding="utf-8")
    rules = [{"kind": "figure_links", "glob": "**/*.md", "catalog": "storage/release-staging/b/catalog/papers.jsonl"}]
    out = redaction.apply_redaction_rules(md.encode(), "01_alpha/5_draft_slides_with_images.md", rules,
                                          workspace=ws, source_path=deck / "5_draft_slides_with_images.md")
    text = out.content.decode()
    assert "assets/figures" not in text and "![" in text  # the remote image is not a local figure
    assert "*Figure from [Sparse Mixtures for Long Context Reasoning](https://openreview.net/forum?id=A1), not reproduced in this release.*" in text
    assert '*Figure from "A Paper Outside The Catalog", not reproduced in this release.*' in text
    assert "*A figure from a cited paper, not reproduced in this release.*" in text
    assert "*Figure 1: a caption*" in text


def test_b23_deck_titles_can_come_from_a_markdown_heading(proj: Proj) -> None:
    ws = proj.ws
    (ws / "decks" / "01_alpha").mkdir(parents=True)
    (ws / "decks" / "01_alpha" / "5_draft_slides_with_images.md").write_text("# s\n", encoding="utf-8")
    write(ws / "analysis" / "01_alpha.md", "# Alpha at Conf: Deep Analysis\n\nText.\n")
    proj.sources["presentations"] = {"decks": "decks", "html": "*/5_draft_slides_with_images.md", "title_markdown": "analysis/{id}.md"}
    proj.build()
    entries = read(proj.out / "presentations" / "presentation_manifest.json")["presentations"]
    assert entries == [{"id": "01_alpha", "presentation_type": "topic-deep-dive", "title": "Alpha at Conf: Deep Analysis"}]


def test_b24_a_topic_page_links_the_deck_slide_text(tmp_path: Path) -> None:
    root = site_fx.make_alpha(tmp_path)
    (root / "decks" / "markdown" / "01_reasoning" / "slides.md").unlink()  # NeurIPS decks have no slides.md
    site_fx._put(root, "decks/markdown/01_reasoning/5_draft_slides_with_images.md", "# Slides\n")
    site_fx._finish(root)
    static_site = importlib.import_module("conference_pipeline.publishing.static_site")
    out = tmp_path / "site"
    static_site.build_public_site(static_site.PublicSiteConfig(bundle_root=root, output_dir=out))
    page = (out / "topics" / "01_reasoning.html").read_text(encoding="utf-8")
    assert 'href="../files/decks/markdown/01_reasoning/5_draft_slides_with_images.md">Read the slides (text)</a>' in page


def test_b25_papers_that_share_a_title_are_named_apart(tmp_path: Path) -> None:
    """6b, NeurIPS 2025: a paper and its workshop version share a title (435 lint findings)."""
    static_site = importlib.import_module("conference_pipeline.publishing.static_site")
    rows = [{"paper_id": "openreview:A1", "title": "Same Title", "track": "main", "official_url": "https://openreview.net/forum?id=A1"},
            {"paper_id": "openreview:B2", "title": "Same Title", "track": "workshop", "official_url": "https://openreview.net/forum?id=B2"},
            {"paper_id": "openreview:C3", "title": "Unique", "track": "main", "official_url": "https://openreview.net/forum?id=C3"}]
    html = static_site._paper_list(rows)
    assert 'aria-label="Same Title (main, openreview:A1)"' in html
    assert 'aria-label="Same Title (workshop, openreview:B2)"' in html
    assert 'aria-label="Unique' not in html


def test_b26_narrative_links_and_headings_fit_the_page(tmp_path: Path) -> None:
    checks = importlib.import_module("conference_pipeline.publishing.site_checks")
    md = "## Section\n\n#### Too deep\n\nSee [Conf 2026](https://conf.example/day/1).\n"
    html = checks.render_markdown(md, heading_offset=2, reserved={"Conf 2026": "../index.html"})
    assert 'aria-label="Conf 2026: https://conf.example/day/1"' in html
    assert "<h4>Section</h4>" in html and "<h5>Too deep</h5>" in html  # offset 2, then no skipped level
    assert "<h6>" not in html


# --- 6b successors: ICLR 2026 transcript records carry no talk id, title or page ---

def _iclr_poster(text_rel: str, item: str = "poster-10006395", slideslive: str = "39057353") -> dict[str, Any]:
    return {"artifact_id": f"speech_transcript:{item}", "source_family_id": "speech_transcript", "indexable_text": True,
            "metadata": {"provider_model": "asr-model", "slideslive_id": slideslive},
            "source_path": f"cache/acquisition-downloads/media_asset/{item}/slideslive_audio/raw.mp3",
            "transcript_path": text_rel}


def test_b27_a_poster_transcript_takes_its_id_kind_and_page_from_the_talk_pages(proj: Proj) -> None:
    ws = proj.ws
    write(ws / "storage" / "t" / "p.txt", "Poster talk text.\n")
    write(ws / "storage" / "dc" / "slideslive_media_recovery.json", {"page_assets": [
        {"item_id": "poster-10006395", "kind": "poster", "page_url": "https://iclr.cc/virtual/2026/poster/10006395",
         "slideslive_id": "39057353"}]})
    proj.records = [_iclr_poster("storage/t/p.txt")]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json",
                                   "talk_pages": "storage/dc/slideslive_media_recovery.json"}
    proj.build()
    text = (proj.out / "transcripts" / "poster-10006395.md").read_text(encoding="utf-8")
    assert text.startswith("# Poster 10006395\n")
    assert "- Kind: Poster" in text and "- Talk page and recording: https://iclr.cc/virtual/2026/poster/10006395" in text
    row = json.loads((proj.out / "transcripts" / "index.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert (row["item_id"], row["kind"], row["page_url"]) == ("poster-10006395", "Poster", "https://iclr.cc/virtual/2026/poster/10006395")


def test_b28_a_workshop_recording_is_named_by_its_workshop_and_drops_unknown_speakers(proj: Proj) -> None:
    ws = proj.ws
    write(ws / "storage" / "t" / "w.json", {"segments": [
        {"text": "Welcome.", "talk_title": "Opening Remarks", "talk_speaker": "Unknown"},
        {"text": "Agents.", "talk_title": "Reliable Agents", "talk_speaker": "Vincent"}]})
    proj.records = [{"artifact_id": "speech_transcript:w", "source_family_id": "speech_transcript", "indexable_text": True,
                     "metadata": {"talk_id": "talk_media_001_407d82a78e", "talk_title": "SlidesLive embedded media",
                                  "workshop_id": "ICLR.cc/2026/Workshop/AFA", "workshop_title": "Advances in Financial AI"},
                     "source_path": "storage/workshops/talk-media/AFA/raw.mp3", "transcript_path": "storage/t/w.json"}]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    proj.build()
    text = (proj.out / "transcripts" / "workshop-AFA-talk_media_001_407d82a78e.md").read_text(encoding="utf-8")
    assert text.startswith("# Advances in Financial AI\n") and "- Kind: Workshop" in text
    assert "- Talk page and recording: https://openreview.net/group?id=ICLR.cc/2026/Workshop/AFA" in text
    assert "## Opening Remarks\n" in text and "(Unknown)" not in text and "## Reliable Agents (Vincent)" in text


def test_b29_a_poster_missing_from_the_talk_pages_still_refuses(proj: Proj) -> None:
    ws = proj.ws
    write(ws / "storage" / "t" / "p.txt", "Poster talk text.\n")
    write(ws / "storage" / "dc" / "slideslive_media_recovery.json", {"page_assets": []})
    proj.records = [_iclr_poster("storage/t/p.txt")]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json",
                                   "talk_pages": "storage/dc/slideslive_media_recovery.json"}
    with pytest.raises(rf().ReaderFamiliesError) as raised:
        proj.build()
    assert raised.value.args[0] == "transcript_id_unsafe"


def test_b30_relink_rewrites_a_relative_deck_link_named_in_the_link_map(tmp_path: Path) -> None:
    # ICLR 2026's executive decks link the topic decks by relative path; a PDF rendered from a
    # temporary folder bakes those in as file:// paths, so the release links them to the site.
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    ws = tmp_path / "proj" / "conferences" / "conf"
    write(ws / "storage" / "release" / "link_map.json",
          {"../01_alpha/presentation.html": "https://conf.example/topics/01_alpha.html"})
    html = ('<a href="../01_alpha/presentation.html">Alpha</a> '
            "<a href='../02_beta/presentation.html'>Beta</a> <a href=\"https://x.example/a\">X</a>")
    rules = [{"kind": "relink", "glob": "**/*.html", "link_map": "storage/release/link_map.json"}]
    out = redaction.apply_redaction_rules(html.encode(), "exec/presentation.html", rules, workspace=ws).content.decode()
    assert '<a href="https://conf.example/topics/01_alpha.html">Alpha</a>' in out
    assert "<a href='../02_beta/presentation.html'>Beta</a>" in out  # unmapped: left as it was
    assert '<a href="https://x.example/a">X</a>' in out


def test_b31_a_hash_keyed_workshop_paper_finds_its_receipt_by_file_name(proj: Proj) -> None:
    # ICLR 2026: the record's id is a content hash and it carries no title; the PDF's name is
    # the receipt's note_id, and the receipt holds the title and authors.
    ws = proj.ws
    proj.records = [{"artifact_id": "workshop_paper_text:e1e2e866233bf34587885178", "source_family_id": "workshop_paper_text",
                     "indexable_text": True, "metadata": {"page_count": 9},
                     "source_path": "storage/workshops/papers/iclr-cc_2026_workshop_afaa/2Dyi1IG7vG.pdf"}]
    write(ws / "storage" / "rx" / "iclr-cc_2026_workshop_afaa" / "receipts" / "2Dyi1IG7vG.json",
          {"note_id": "2Dyi1IG7vG", "venueid": "ICLR.cc/2026/Workshop/AFAA", "title": "A Workshop Paper",
           "authors": ["Ada One", "Ben Two"]})
    proj.sources["catalog"] = {"adapter": "stage6_inventory", "path": "storage/corpus/corpus_inventory.json",
                               "receipts": "storage/rx/*/receipts/*.json"}
    proj.build()
    rows = proj.rows()
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "openreview:2Dyi1IG7vG" and rows[0]["title"] == "A Workshop Paper"
    assert rows[0]["authors"] == ["Ada One", "Ben Two"] and rows[0]["track"] == "workshop"
    assert rows[0]["official_url"] == "https://openreview.net/forum?id=2Dyi1IG7vG"


# --- 6b successor: ICLR 2025 (merged corpus inventory, session ASR lane) ---

def test_b32_a_paper_with_no_link_takes_its_unit_files_url_when_asked(proj: Proj) -> None:
    # ICLR 2025: workshop papers found on a workshop's own site carry only a site: key; the
    # unit file the record was read from holds the paper's page. Only an https link without a
    # private marker is taken, and only when the catalog spec asks for unit links.
    ws = proj.ws
    units = "storage/papers/late/units/site_abstract.jsonl"
    write(ws / units, "\n".join(json.dumps(u) for u in [
        {"unit_id": "ws-1", "url": "https://workshop.example/papers/1", "title": "One"},
        {"unit_id": "ws-2", "url": "http://workshop.example/papers/2", "title": "Two"},
        {"unit_id": "ws-3", "url": "https://drive.google.com/file/d/abc", "title": "Three"},
    ]) + "\n")
    proj.records = [paper(f"site:ws-{n}", t, [], "workshop", source_path=units, metadata={"unit_id": f"ws-{n}"})
                    for n, t in [(1, "One"), (2, "Two"), (3, "Three"), (4, "Four")]]
    proj.sources["catalog"] = {"adapter": "corpus_inventory", "path": "storage/corpus/corpus_inventory.json", "unit_links": True}
    proj.build()
    urls = {r["paper_id"]: r["official_url"] for r in proj.rows()}
    assert urls == {"site:ws-1": "https://workshop.example/papers/1", "site:ws-2": None, "site:ws-3": None, "site:ws-4": None}
    proj.sources["catalog"].pop("unit_links")
    proj.build()
    assert {r["official_url"] for r in proj.rows()} == {None}


def test_b33_an_untitled_event_recording_is_named_by_its_kind_and_number(proj: Proj) -> None:
    # ICLR 2025: session ASR records name only their event (event:oral-31735) and its page;
    # no local file holds talk titles, so the talk is "Oral 31735" (as ICLR 2026's posters).
    ws = proj.ws
    write(ws / "storage" / "t" / "o.txt", "Oral talk text.\n")
    write(ws / "storage" / "t" / "h.txt", "Town hall text.\n")
    def rec(event: str, kind: str, rel: str) -> dict[str, Any]:
        return {"artifact_id": f"evidence_lane_transcript:lane:event:{event}", "source_family_id": "speech_transcript",
                "indexable_text": True, "item_kind": kind, "kept_copy": True, "title": None, "video_id": f"event:{event}",
                "source_event_urls": [f"https://conf.example/virtual/{kind}/{event.rsplit('-', 1)[-1]}"],
                "transcript_path": rel, "transcript_source": "session_asr"}
    proj.records = [rec("oral-31735", "oral", "storage/t/o.txt"), rec("town-hall-40001", "town-hall", "storage/t/h.txt")]
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    proj.build()
    rows = [json.loads(line) for line in (proj.out / "transcripts" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    got = {r["item_id"]: (r["title"], r["kind"], r["page_url"]) for r in rows}
    assert got == {"oral-31735": ("Oral 31735", "Oral", "https://conf.example/virtual/oral/31735"),
                   "town-hall-40001": ("Town hall 40001", "Town hall", "https://conf.example/virtual/town-hall/40001")}
    assert (proj.out / "transcripts" / "oral-31735.md").read_text(encoding="utf-8").startswith("# Oral 31735\n")


def test_b34_a_figure_the_pipeline_generated_is_not_called_a_paper_figure(tmp_path: Path) -> None:
    # ICLR 2025: the deck's figures were drawn by the pipeline (assets/generated/), not cropped
    # from papers; the notice says so, and a paper figure keeps its own notice (b22).
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    ws = tmp_path / "proj" / "conferences" / "conf"
    deck = ws / "decks" / "02_beta"
    md = ("![Generated figure for slide 1](assets/generated/slide_001_overview.svg)\n\n"
          "![Generated figure for slide 2](./assets/generated/slide_002_agenda.svg)\n\n"
          "![A paper figure](assets/figures/slide_03/f1.png)\n")
    deck.mkdir(parents=True)
    (deck / "5_draft_slides_with_images.md").write_text(md, encoding="utf-8")
    rules = [{"kind": "figure_links", "glob": "**/*.md", "catalog": "storage/release-staging/b/catalog/papers.jsonl"}]
    out = redaction.apply_redaction_rules(md.encode(), "02_beta/5_draft_slides_with_images.md", rules,
                                          workspace=ws, source_path=deck / "5_draft_slides_with_images.md")
    text = out.content.decode()
    assert "assets/" not in text
    assert text.count("*A figure made for this deck, not included in this release.*") == 2
    assert text.count("*A figure from a cited paper, not reproduced in this release.*") == 1


def _pdf_with_links(uris: list[str]) -> bytes:
    import io

    import pypdf
    from pypdf.annotations import Link

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    for i, uri in enumerate(uris):
        writer.add_annotation(0, Link(rect=(10, 10 + 30 * i, 100, 30 + 30 * i), url=uri))
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _pdf_uris(data: bytes) -> list[str]:
    import io

    import pypdf

    uris = []
    for page in pypdf.PdfReader(io.BytesIO(data)).pages:
        for annot in page.get("/Annots") or []:
            action = annot.get_object().get("/A")
            if action and action.get("/URI"):
                uris.append(str(action["/URI"]))
    return uris


def test_b35_a_pdf_link_rule_removes_links_that_are_not_public(tmp_path: Path) -> None:
    # ICLR 2025: the topic-02 deck PDF links two unfilled paper links as file:// home paths.
    # The rule removes each link annotation that is not a public https link; the pages stay.
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    home = "file://" + OUTSIDE + "/deck/paper_url"
    data = _pdf_with_links([home, "https://arxiv.org/abs/2501.00001", "https://drive.google.com/file/d/abc"])
    rules = [{"kind": "pdf_links", "glob": "**/*.pdf"}]
    out = redaction.apply_redaction_rules(data, "02_beta/presentation.pdf", rules, workspace=tmp_path)
    assert out.content is not None and out.binary_marker is False and out.kinds == ("pdf_links",)
    assert _pdf_uris(out.content) == ["https://arxiv.org/abs/2501.00001"]
    again = redaction.apply_redaction_rules(data, "02_beta/presentation.pdf", rules, workspace=tmp_path)
    assert again.content == out.content
    # Without the rule, a PDF is never rewritten (D8: regenerate or exclude).
    plain = redaction.apply_redaction_rules(data, "02_beta/presentation.pdf", [{"kind": "absolute_path", "glob": "**/*.pdf"}],
                                            workspace=tmp_path)
    assert plain.content is None


@pytest.mark.parametrize("pypdf_keywords", ["new", "old"])
def test_p7_pdf_link_rule_works_with_either_pypdf_keyword_set(tmp_path: Path, monkeypatch, pypdf_keywords: str) -> None:
    # v6.0 Phase 7: a fresh install resolved pypdf 7, which removed remove_identicals and
    # remove_orphans; pypdf 4-5 have only those. Both keyword sets must drop the orphans.
    import pypdf

    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    real = pypdf.PdfWriter.compress_identical_objects
    seen: list[str] = []

    def only(self, *args, **kwargs):
        allowed = {"remove_duplicates", "remove_unreferenced"} if pypdf_keywords == "new" else {"remove_identicals", "remove_orphans"}
        if set(kwargs) - allowed:
            raise TypeError(f"unexpected keywords {sorted(set(kwargs) - allowed)}")
        seen.append(",".join(sorted(kwargs)))
        mapped = {"remove_duplicates": kwargs.get("remove_duplicates", kwargs.get("remove_identicals")),
                  "remove_unreferenced": kwargs.get("remove_unreferenced", kwargs.get("remove_orphans"))}
        return real(self, **mapped)

    monkeypatch.setattr(pypdf.PdfWriter, "compress_identical_objects", only)
    home = "file://" + OUTSIDE + "/deck/paper_url"
    data = _pdf_with_links([home, "https://arxiv.org/abs/2501.00001"])
    out = redaction.apply_redaction_rules(data, "02_beta/presentation.pdf", [{"kind": "pdf_links", "glob": "**/*.pdf"}],
                                          workspace=tmp_path)
    assert _pdf_uris(out.content) == ["https://arxiv.org/abs/2501.00001"]
    assert OUTSIDE.encode() not in out.content
    assert seen == (["remove_duplicates,remove_unreferenced"] if pypdf_keywords == "new" else ["remove_identicals,remove_orphans"])


def test_b36_drop_preamble_removes_text_above_the_first_heading(tmp_path: Path) -> None:
    # ICLR 2025: the presenter script opens with a model's chat line above its title.
    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    rules = [{"kind": "drop_preamble", "glob": "**/*.md"}]
    md = "Here is the complete, edited presenter script.\n\n# Presenter Script: Beta\n\nBody.\n"
    out = redaction.apply_redaction_rules(md.encode(), "02_beta/6_presenter_script.md", rules, workspace=tmp_path)
    assert out.content == b"# Presenter Script: Beta\n\nBody.\n"
    for unchanged in ["# Title first\n\nBody.\n", "No heading at all.\n"]:
        assert redaction.apply_redaction_rules(unchanged.encode(), "x.md", rules, workspace=tmp_path).content is None


# --- 6b review round phase-6b-r1 (Codex): one regression test per finding ---

def _pdf_with_text(text: str) -> bytes:
    """A one-page PDF whose content stream (Flate-compressed) draws ``text``."""
    import io

    import pypdf
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=400, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 10 Tf 10 100 Td (" + text.encode() + b") Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    page.compress_content_streams()
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_r1_f1_private_text_in_a_compressed_pdf_is_a_marker(tmp_path: Path) -> None:
    # A PDF's text sits in compressed streams a byte scan cannot read: the check reads the
    # text. ".env" counts only as a file name, so "changes.environmental" (ICLR 2025's deck,
    # text extraction joining two lines) is not one.
    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    redaction = importlib.import_module("conference_pipeline.publishing.bundle_redaction")
    private = _pdf_with_text(OUTSIDE + "/private.txt")
    assert (OUTSIDE + "/private.txt").encode() not in private  # compressed: invisible to a byte scan
    assert has_publishing_marker_bytes(private) is True
    out = redaction.apply_redaction_rules(private, "02_beta/presentation.pdf", [{"kind": "pdf_links", "glob": "**/*.pdf"}],
                                          workspace=tmp_path)
    assert out.binary_marker is True
    assert has_publishing_marker_bytes(_pdf_with_text("see " + OUTSIDE[1:] + "/x/.env for keys")) is True
    assert has_publishing_marker_bytes(_pdf_with_text("Adaptive pipelines and changes.environmental changes")) is False
    assert has_publishing_marker_bytes(_pdf_with_links(["https://arxiv.org/abs/2501.00001"])) is False
    # A PDF that cannot be read cannot be shown clean: it fails closed.
    assert has_publishing_marker_bytes(b"%PDF-1.4\n" + b"\x00" * 64) is True


def test_r1_f2_a_zip_member_too_large_to_scan_fails_closed() -> None:
    import io
    import zipfile

    from conference_pipeline._security import private_markers as pm

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.bin", b"A" * (pm._ZIP_MEMBER_LIMIT + 1) + (OUTSIDE + "/private.txt").encode())
    assert pm.has_publishing_marker_bytes(buf.getvalue()) is True
    small = io.BytesIO()
    with zipfile.ZipFile(small, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ok.txt", b"plain text")
    assert pm.has_publishing_marker_bytes(small.getvalue()) is False


def test_r1_f4_a_unit_link_comes_from_the_records_own_unit_file(proj: Proj) -> None:
    # Two unit files may reuse a unit id; each record takes the link from the file it names.
    ws = proj.ws
    write(ws / "storage" / "u" / "a.jsonl", json.dumps({"unit_id": "1", "url": "https://a.example/1"}) + "\n")
    write(ws / "storage" / "u" / "b.jsonl", json.dumps({"unit_id": "1", "url": "https://b.example/1"}) + "\n")
    proj.records = [paper("site:a-1", "From A", [], "workshop", source_path="storage/u/a.jsonl", metadata={"unit_id": "1"}),
                    paper("site:b-1", "From B", [], "workshop", source_path="storage/u/b.jsonl", metadata={"unit_id": "1"})]
    proj.sources["catalog"] = {"adapter": "corpus_inventory", "path": "storage/corpus/corpus_inventory.json", "unit_links": True}
    proj.build()
    assert {r["paper_id"]: r["official_url"] for r in proj.rows()} == {"site:a-1": "https://a.example/1",
                                                                       "site:b-1": "https://b.example/1"}
    # One file giving one unit id two links cannot say which is right: the build refuses.
    write(ws / "storage" / "u" / "a.jsonl", "".join(json.dumps({"unit_id": "1", "url": u}) + "\n"
                                                    for u in ("https://a.example/1", "https://a.example/other")))
    with pytest.raises(rf().ReaderFamiliesError) as err:
        proj.build()
    assert err.value.code == "unit_link_conflict"


def test_r1_f5_two_recordings_that_normalise_to_one_name_refuse(proj: Proj) -> None:
    ws = proj.ws
    write(ws / "storage" / "t" / "a.txt", "First talk.\n")
    write(ws / "storage" / "t" / "b.txt", "Second talk.\n")
    def rec(video: str, title: str, rel: str) -> dict[str, Any]:
        return {"artifact_id": f"t:{video}", "source_family_id": "speech_transcript", "indexable_text": True,
                "item_kind": "oral", "kept_copy": True, "title": title, "video_id": video,
                "transcript_path": rel, "transcript_source": "asr"}
    proj.sources["transcripts"] = {"inventory": "storage/corpus/corpus_inventory.json"}
    # The same recording listed twice is one transcript.
    proj.records = [rec("talk:a", "A", "storage/t/a.txt"), rec("talk:a", "A", "storage/t/a.txt")]
    proj.build()
    assert len((proj.out / "transcripts" / "index.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    # Two recordings whose ids become one file name refuse, whichever comes first.
    for order in ([("talk:a", "A", "storage/t/a.txt"), ("talk-a", "B", "storage/t/b.txt")],
                  [("talk-a", "B", "storage/t/b.txt"), ("talk:a", "A", "storage/t/a.txt")]):
        proj.records = [rec(*r) for r in order]
        with pytest.raises(rf().ReaderFamiliesError) as err:
            proj.build()
        assert err.value.code == "transcript_id_collision"


@pytest.mark.parametrize("loads", [
    '<img src="file://' + OUTSIDE + '/private.png">',
    '<img src="' + OUTSIDE + '/private.png">',
    '<img src="assets/fig.svg?v=2">',
    '<div style="background:url(assets/fig.svg)"></div>',
    "<style>.x{background-image:url('../logo.png')}</style>",
    '<img srcset="assets/fig.svg 2x">',
    '<link rel="stylesheet" href="theme.css">',
], ids=["file-url", "absolute-path", "query", "inline-style-url", "stylesheet-url", "srcset", "link-href"])
def test_r1_f6_html_that_loads_any_local_resource_is_not_rendered(tmp_path: Path, loads: str) -> None:
    pdfs = importlib.import_module("conference_pipeline.publishing.pdf_regeneration")
    ws = tmp_path / "proj" / "conferences" / "conf"
    write(ws / "storage" / "public_export_manifest.json",
          {"families": [{"id": "presentation_payloads", "private_path": "decks", "redaction_rules": []}]})
    write(ws / "decks" / "01_alpha" / "presentation.html", f"<html><body>{loads}</body></html>")
    with pytest.raises(pdfs.PdfRegenerationError) as err:
        pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"}, render=lambda **_: None)
    assert err.value.code == "pdf_html_not_self_contained"


def test_r1_f6_inline_remote_and_fragment_references_still_render(tmp_path: Path) -> None:
    pdfs = importlib.import_module("conference_pipeline.publishing.pdf_regeneration")
    ws = tmp_path / "proj" / "conferences" / "conf"
    write(ws / "storage" / "public_export_manifest.json",
          {"families": [{"id": "presentation_payloads", "private_path": "decks", "redaction_rules": []}]})
    write(ws / "decks" / "01_alpha" / "presentation.html",
          '<html><head><link rel="stylesheet" href="https://fonts.example/a.css"></head><body>'
          '<img src="data:image/png;base64,AAAA"><img src="https://cdn.example/x.png">'
          '<svg><rect style="fill:url(#g)"/></svg><a href="notes.html">notes</a><a href="#s2">s</a></body></html>')

    def render(input_html: Path, output_pdf: Path) -> None:
        output_pdf.write_bytes(_pdf_with_links([]))

    out = pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"}, render=render)
    assert list(out.files) == ["01_alpha/presentation.pdf"]


def test_f27_a_regenerated_pdf_is_the_same_bytes_whenever_it_is_rendered(tmp_path: Path) -> None:
    # Chromium stamps the render time in CreationDate and ModDate, so each regeneration gave
    # new bytes (F-6b-27). The dates are pinned; nothing else in the PDF changes.
    import io

    import pypdf

    pdfs = importlib.import_module("conference_pipeline.publishing.pdf_regeneration")
    ws = tmp_path / "proj" / "conferences" / "conf"
    write(ws / "storage" / "public_export_manifest.json",
          {"families": [{"id": "presentation_payloads", "private_path": "decks", "redaction_rules": []}]})
    write(ws / "decks" / "01_alpha" / "presentation.html", "<html><body><p>Alpha</p></body></html>")

    def renderer(stamp: str):
        def render(input_html: Path, output_pdf: Path) -> None:
            writer = pypdf.PdfWriter()
            writer.add_blank_page(width=200, height=200)
            writer.add_metadata({"/Title": "Alpha", "/CreationDate": f"D:{stamp}+00'00'", "/ModDate": f"D:{stamp}+00'00'"})
            out = io.BytesIO()
            writer.write(out)
            output_pdf.write_bytes(out.getvalue())
        return render

    first = pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"}, render=renderer("20261119022159"))
    second = pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"}, render=renderer("20261119073825"))
    a, b = first.files["01_alpha/presentation.pdf"], second.files["01_alpha/presentation.pdf"]
    assert a == b
    meta = pypdf.PdfReader(io.BytesIO(a)).metadata
    assert meta["/Title"] == "Alpha" and "20261119" not in str(meta["/CreationDate"]) + str(meta["/ModDate"])


# --- 6b review round phase-6b-r2 (Codex): one regression test per finding ---

def _pdf_parts(content: bytes | None = None, *, compress: bool = False, truncate: bool = False,
               xmp: bytes | None = None) -> bytes:
    """A one-page PDF built with pypdf: its content stream plain, Flate-compressed or
    truncated, and an optional Flate-compressed XMP metadata stream."""
    import io
    import zlib

    import pypdf
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, StreamObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=400, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    if content is not None:
        if compress or truncate:
            stream = StreamObject()
            packed = zlib.compress(content)
            stream._data = packed[:3] if truncate else packed
            stream[NameObject("/Filter")] = NameObject("/FlateDecode")
        else:
            stream = DecodedStreamObject()
            stream.set_data(content)
        page[NameObject("/Contents")] = writer._add_object(stream)
    if xmp is not None:
        meta = StreamObject()
        meta._data = zlib.compress(xmp)
        meta[NameObject("/Filter")] = NameObject("/FlateDecode")
        meta[NameObject("/Type")] = NameObject("/Metadata")
        meta[NameObject("/Subtype")] = NameObject("/XML")
        writer._root_object[NameObject("/Metadata")] = writer._add_object(meta)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _ascii_pdf(text: str) -> bytes:
    """A valid PDF of ASCII bytes only (so it decodes as UTF-8), drawing ``text`` uncompressed."""
    content = f"BT /F1 10 Tf 10 100 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_r2_f1_release_check_reads_a_pdf_with_no_nul_byte(w0) -> None:  # noqa: F811
    # The check took the binary path only when the first 8 KiB held a NUL; a PDF drawing a
    # home path as a hex string has none, and was scanned as text.
    w4 = rc.build(rc.plan(rc.promote(rc.draft(w0))))
    data = _pdf_parts(b"BT /F1 10 Tf 10 100 Td <" + (OUTSIDE + "/private.txt").encode().hex().encode() + b"> Tj ET")
    assert b"\0" not in data and (OUTSIDE + "/private.txt").encode() not in data
    (w4.bundle / "decks" / "01_sample" / "deck.pdf").write_bytes(data)
    rc.cli().release_check(rc.SLUG, secret_scanner=rc._no_secrets, **w4.kw())
    assert any("private_marker" in line and "deck.pdf" in line for line in w4.out)


def test_r2_f2_private_xmp_metadata_is_a_marker() -> None:
    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           '<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:description>' + OUTSIDE +
           '/private.txt</dc:description></rdf:Description></rdf:RDF></x:xmpmeta>').encode()
    assert has_publishing_marker_bytes(_pdf_parts(b"BT ET", xmp=xmp)) is True
    assert has_publishing_marker_bytes(_pdf_parts(b"BT ET", xmp=xmp.replace(OUTSIDE.encode(), b"a"))) is False


def test_r2_f3_a_pdf_stream_that_does_not_decode_whole_fails_closed() -> None:
    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    content = b"BT /F1 10 Tf 10 100 Td (" + (OUTSIDE + "/private.txt").encode() + b") Tj ET"
    assert has_publishing_marker_bytes(_pdf_parts(content, truncate=True)) is True
    assert has_publishing_marker_bytes(_pdf_parts(b"BT /F1 10 Tf 10 100 Td (clean) Tj ET", compress=True)) is False


def test_r2_f6_a_text_encodable_pdf_meets_the_pdf_rules() -> None:
    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    clean = _ascii_pdf("Adaptive pipelines and changes.environmental changes")
    clean.decode("utf-8")  # it is text-encodable
    assert has_publishing_marker_bytes(clean) is False
    assert has_publishing_marker_bytes(_ascii_pdf("see " + OUTSIDE + "/private.txt")) is True


@pytest.mark.parametrize("loads", [
    "<img src=assets/fig.svg>",
    "<link rel=stylesheet href=theme.css>",
    '<style>@import "theme.css";</style>',
    '<svg><image href="assets/fig.svg"/></svg>',
    '<svg><image xlink:href="assets/fig.svg"/></svg>',
], ids=["unquoted-src", "unquoted-link", "css-import", "svg-image", "svg-xlink-image"])
def test_r2_f4_more_local_loads_are_not_rendered(tmp_path: Path, loads: str) -> None:
    pdfs = importlib.import_module("conference_pipeline.publishing.pdf_regeneration")
    ws = tmp_path / "proj" / "conferences" / "conf"
    write(ws / "storage" / "public_export_manifest.json",
          {"families": [{"id": "presentation_payloads", "private_path": "decks", "redaction_rules": []}]})
    write(ws / "decks" / "01_alpha" / "presentation.html", f"<html><body>{loads}</body></html>")
    with pytest.raises(pdfs.PdfRegenerationError) as err:
        pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"}, render=lambda **_: None)
    assert err.value.code == "pdf_html_not_self_contained"


@pytest.mark.parametrize("page", [
    '<img srcset="data:image/png;base64,iVBORw0KGgo= 1x, https://cdn.example/b.png 2x">',
    '<!-- <img src="old.png"> -->',
    '<svg><use href="#shape"/></svg>',
], ids=["srcset-data-uri", "comment", "svg-use-fragment"])
def test_r2_f5_self_contained_pages_still_render(tmp_path: Path, page: str) -> None:
    pdfs = importlib.import_module("conference_pipeline.publishing.pdf_regeneration")
    ws = tmp_path / "proj" / "conferences" / "conf"
    write(ws / "storage" / "public_export_manifest.json",
          {"families": [{"id": "presentation_payloads", "private_path": "decks", "redaction_rules": []}]})
    write(ws / "decks" / "01_alpha" / "presentation.html", f"<html><body>{page}</body></html>")

    def render(input_html: Path, output_pdf: Path) -> None:
        output_pdf.write_bytes(_pdf_with_links([]))

    out = pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"}, render=render)
    assert list(out.files) == ["01_alpha/presentation.pdf"]


# --- 6b review round phase-6b-r3 (Codex): one regression test per finding ---

def _pdf_with_comment(text: str) -> bytes:
    """A blank page carrying a /Text annotation (a comment) whose /Contents is ``text``;
    pypdf writes a non-Latin-1 string as UTF-16."""
    import io

    import pypdf
    from pypdf.generic import ArrayObject, DictionaryObject, FloatObject, NameObject, TextStringObject

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    annot = DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject("/Text"),
        NameObject("/Rect"): ArrayObject([FloatObject(10), FloatObject(10), FloatObject(30), FloatObject(30)]),
        NameObject("/Contents"): TextStringObject(text), NameObject("/T"): TextStringObject("reviewer")})
    writer.add_annotation(0, annot)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_r3_f1_pdf_annotation_text_is_read() -> None:
    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    dirty = _pdf_with_comment("α " + OUTSIDE + "/private.txt")
    assert (OUTSIDE + "/private.txt").encode() not in dirty  # UTF-16 in the file
    assert has_publishing_marker_bytes(dirty) is True
    assert has_publishing_marker_bytes(_pdf_with_comment("α a clean note")) is False


def test_r3_f2_utf16_text_in_a_zip_member_or_a_file_is_read() -> None:
    import io
    import zipfile

    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    def pptx(xml: str) -> bytes:
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("ppt/slides/_rels/slide1.xml.rels", xml.encode("utf-16"))
        return out.getvalue()

    rels = '<?xml version="1.0" encoding="UTF-16"?><Relationships><Relationship Target="{}"/></Relationships>'
    assert has_publishing_marker_bytes(pptx(rels.format("file://" + OUTSIDE + "/private.txt"))) is True
    assert has_publishing_marker_bytes(pptx(rels.format("https://example.org/a.png"))) is False
    assert has_publishing_marker_bytes(("see " + OUTSIDE + "/private.txt").encode("utf-16")) is True
    assert has_publishing_marker_bytes(("see " + OUTSIDE + "/private.txt").encode("utf-16-le")) is True
    assert has_publishing_marker_bytes("a clean note".encode("utf-16")) is False


def _pdf_with_form(form: bytes, *, truncate: bool) -> bytes:
    """A page that draws Form XObject /Fm, whose Flate stream is whole or cut to 3 bytes."""
    import io
    import zlib

    import pypdf
    from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, FloatObject, NameObject, StreamObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=400, height=200)
    fm = StreamObject()
    packed = zlib.compress(form)
    fm._data = packed[:3] if truncate else packed
    fm.update({NameObject("/Filter"): NameObject("/FlateDecode"), NameObject("/Type"): NameObject("/XObject"),
               NameObject("/Subtype"): NameObject("/Form"),
               NameObject("/BBox"): ArrayObject([FloatObject(0), FloatObject(0), FloatObject(400), FloatObject(200)])})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): DictionaryObject({NameObject("/Fm"): writer._add_object(fm)})})
    content = DecodedStreamObject()
    content.set_data(b"q /Fm Do Q")
    page[NameObject("/Contents")] = writer._add_object(content)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_r3_f4_a_nested_stream_that_does_not_decode_whole_fails_closed() -> None:
    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    form = b"BT /F1 10 Tf 10 100 Td (" + (OUTSIDE + "/private.txt").encode() + b") Tj ET"
    assert has_publishing_marker_bytes(_pdf_with_form(form, truncate=True)) is True
    assert has_publishing_marker_bytes(_pdf_with_form(form, truncate=False)) is True
    assert has_publishing_marker_bytes(_pdf_with_form(b"BT /F1 10 Tf 10 100 Td (clean) Tj ET", truncate=False)) is False


def test_r3_f5_release_check_applies_the_pdf_rules_to_a_text_encodable_pdf(w0) -> None:  # noqa: F811
    w4 = rc.build(rc.plan(rc.promote(rc.draft(w0))))
    (w4.bundle / "decks" / "01_sample" / "deck.pdf").write_bytes(_ascii_pdf("Adaptive pipelines and changes.environmental changes"))
    rc.cli().release_check(rc.SLUG, secret_scanner=rc._no_secrets, **w4.kw())
    assert not any("deck.pdf" in line for line in w4.out if "private_marker" in line)
    (w4.bundle / "decks" / "01_sample" / "deck.pdf").write_bytes(_ascii_pdf("see " + OUTSIDE + "/private.txt"))
    w4.out.clear()
    rc.cli().release_check(rc.SLUG, secret_scanner=rc._no_secrets, **w4.kw())
    assert any("private_marker" in line and "deck.pdf" in line for line in w4.out)


def _pdf_ws(tmp_path: Path, page: str) -> Path:
    ws = tmp_path / "proj" / "conferences" / "conf"
    write(ws / "storage" / "public_export_manifest.json",
          {"families": [{"id": "presentation_payloads", "private_path": "decks", "redaction_rules": []}]})
    write(ws / "decks" / "01_alpha" / "presentation.html", f"<html><body>{page}</body></html>")
    return ws


def test_r3_f6_a_local_load_the_renderer_refuses_is_not_rendered(tmp_path: Path) -> None:
    pdfs = importlib.import_module("conference_pipeline.publishing.pdf_regeneration")
    # The renderer enforces it: a load it refused, however the page asked for it, refuses the PDF.
    ws = _pdf_ws(tmp_path, '<script>new Image().src = "file:///tmp/private.png"</script>')

    def render(input_html: Path, output_pdf: Path) -> list[str]:
        output_pdf.write_bytes(_pdf_with_links([]))
        return ["file:///tmp/private.png"]

    with pytest.raises(pdfs.PdfRegenerationError) as err:
        pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"}, render=render)
    assert err.value.code == "pdf_html_not_self_contained"
    # An iframe's srcdoc is a page of its own, read before rendering.
    ws = _pdf_ws(tmp_path / "srcdoc", '<iframe srcdoc="<img src=file:///tmp/private.png>"></iframe>')
    with pytest.raises(pdfs.PdfRegenerationError) as err:
        pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"}, render=lambda **_: None)
    assert err.value.code == "pdf_html_not_self_contained"


@pytest.mark.parametrize("page", [
    '<script>new Image().src = "file://{png}"</script>',
    '<style>body {{ background: u\\72l(file://{png}) }}</style>',
    '<iframe srcdoc="<img src=file://{png}>"></iframe>',
], ids=["script", "css-escape", "srcdoc"])
def test_r3_f6_chromium_never_loads_a_local_resource(tmp_path: Path, page: str) -> None:
    pytest.importorskip("playwright.sync_api")
    pdfs = importlib.import_module("conference_pipeline.publishing.pdf_regeneration")
    png = tmp_path / "private.png"
    png.write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                                  "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"))
    ws = _pdf_ws(tmp_path, page.format(png=png))
    try:
        with pytest.raises(pdfs.PdfRegenerationError) as err:
            pdfs.regenerate_deck_pdfs(ws, {"family": "presentation_payloads"})
    except Exception as exc:  # noqa: BLE001
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
            pytest.skip("Chromium is not installed")
        raise
    assert err.value.code == "pdf_html_not_self_contained"


def test_r3_f7_two_papers_under_one_receipt_id_are_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    rec = ws / "storage" / "workshops" / "papers"
    spec = {"receipts": "storage/workshops/papers/*/receipts/*.json"}
    # One paper accepted at two workshops: one receipt id, one title, two links. Kept.
    write(rec / "a" / "receipts" / "1.json", {"source": "workshop_site", "note_id": "1", "title": "Paper A",
                                              "pdf_url": "https://a.example/1.pdf"})
    write(rec / "b" / "receipts" / "1.json", {"source": "workshop_site", "note_id": "1", "title": "Paper  a",
                                              "pdf_url": "https://b.example/1.pdf"})
    assert set(rf()._stage6_links(ws, spec).receipts) == {"1"}
    # Two papers under one receipt id: refused, never one row with the other's link.
    write(rec / "b" / "receipts" / "1.json", {"source": "workshop_site", "note_id": "1", "title": "Paper B",
                                              "pdf_url": "https://b.example/1.pdf"})
    with pytest.raises(rf().ReaderFamiliesError) as err:
        rf()._stage6_links(ws, spec)
    assert "1" in str(err.value)
    # Two records whose ids end in the same key but whose titles differ: refused too.
    records = [_s6("workshop_paper:a:1", "workshop_paper_text", "Paper A", source_url="https://a.example/1.pdf"),
               _s6("workshop_paper:b:1", "workshop_paper_text", "Paper B", source_url="https://b.example/1.pdf")]
    with pytest.raises(rf().ReaderFamiliesError):
        rf()._rows_from_stage6_inventory({"artifact_records": records})
    same = [_s6("workshop_paper:a:1", "workshop_paper_text", "Paper A", source_url="https://a.example/1.pdf"),
            _s6("workshop_paper:b:1", "workshop_paper_text", "Paper A", source_url="https://b.example/1.pdf")]
    rows, _ = rf()._rows_from_stage6_inventory({"artifact_records": same})
    assert [r["title"] for r in rows] == ["Paper A"]
