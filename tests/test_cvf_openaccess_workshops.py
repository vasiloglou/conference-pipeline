"""Offline coverage for conference_pipeline.acquisition.cvf_openaccess_workshops."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.acquisition import cvf_openaccess_workshops as cvf
from conference_pipeline.acquisition.workshop_pipeline import slugify_venueid

MENU_URL = "https://openaccess.thecvf.com/CONF2099_workshops/menu"

MENU_HTML = """
<div id="content"><h3>Workshops</h3><dl>
<dd><a href="/CONF2099_workshops/EarthVision">EarthVision: Remote Sensing</a><br><br></dd>
<dd><a href="/CONF2099_workshops/6thAdvML@CV">Adversarial ML &amp; CV</a><br><br></dd>
<dd><a href="/CONF2099_workshops/Fresh">Fresh Workshop</a><br><br></dd>
</dl></div>
"""


def _paper_block(stem: str, title: str, authors: list[str], *, arxiv: bool = False) -> str:
    forms = "".join(
        f'<form class="authsearch"><input type="hidden" name="query_author" value="{a}">'
        f'<a href="#">{a}</a>,</form>'
        for a in authors
    )
    links = f'[<a href="/content/CONF2099W/X/papers/{stem}.pdf">pdf</a>]'
    links += f'[<a href="/content/CONF2099W/X/supplemental/{stem}_supp.pdf">supp</a>]'
    if arxiv:
        links += '[<a href="https://arxiv.org/abs/2601.00001">arXiv</a>]'
    return (
        f'<dt class="ptitle"><br><a href="/content/CONF2099W/X/html/{stem}.html">{title}</a></dt>'
        f"<dd>{forms}</dd><dd>{links}</dd>"
    )


def _listing(*blocks: str) -> str:
    return "<dl>" + "".join(blocks) + "</dl>"


def _pdf_bytes(tag: str) -> bytes:
    return b"%PDF-1.7\n" + tag.encode() + b"\n%%EOF"


def test_parse_menu_and_papers() -> None:
    workshops = cvf.parse_workshop_menu(MENU_HTML, MENU_URL)
    assert [w.short_name for w in workshops] == ["EarthVision", "6thAdvML@CV", "Fresh"]
    assert workshops[1].title == "Adversarial ML & CV"
    assert workshops[0].venueid == "openaccess.thecvf.com/CONF2099_workshops/EarthVision"

    page = _listing(
        _paper_block("A_Paper", "Alpha: A &amp; B", ["Ann One", "Bo Two"], arxiv=True),
        _paper_block("B_Paper", "Beta", ["Cy Three"]),
        '<dt class="ptitle"><a href="/content/CONF2099W/X/html/NoPdf.html">No PDF</a></dt>',
    )
    papers = cvf.parse_workshop_papers(page, workshops[0].url)
    assert [p.title for p in papers] == ["Alpha: A & B", "Beta"]
    assert papers[0].authors == ("Ann One", "Bo Two")
    assert papers[0].pdf_url == "https://openaccess.thecvf.com/content/CONF2099W/X/papers/A_Paper.pdf"
    assert papers[0].arxiv_url == "https://arxiv.org/abs/2601.00001"
    assert papers[1].arxiv_url is None
    assert papers[0].paper_id.startswith("cvf_") and papers[0].paper_id != papers[1].paper_id


def test_openreview_venue_matching_is_unambiguous_and_tolerates_ordinals() -> None:
    workshops = cvf.parse_workshop_menu(MENU_HTML, MENU_URL)
    inventory = {
        "workshops": [
            {"venueid": "org/CONF/2099/Workshop/EarthVision", "pdf_availability_class": "no_openreview_record"},
            {"venueid": "org/CONF/2099/Workshop/AdvML@CV", "pdf_availability_class": "openreview_with_pdf"},
            {"venueid": "org/CONF/2099/Workshop/AdvML@CV_Track", "sub_track_parent": "x"},
        ]
    }
    matches = cvf.match_openreview_venues(workshops, inventory)
    assert matches["EarthVision"]["openreview_pdf_availability_class"] == "no_openreview_record"
    assert matches["6thAdvML@CV"]["openreview_venueid"] == "org/CONF/2099/Workshop/AdvML@CV"
    assert "Fresh" not in matches


def test_known_paper_index_catches_retitled_camera_ready_but_not_same_group_new_paper() -> None:
    index = cvf.KnownPaperIndex(
        [
            cvf.KnownPaper(
                title_key=cvf.normalize_title("[UNI]101: An Educational Dataset for Vision"),
                author_keys=frozenset(cvf.author_key(a) for a in ["Ethan Seefried", "Jeremy S Lewis", "Cy Three"]),
                ref="or:1",
            )
        ]
    )
    assert index.find("[UNI]101: an educational dataset, for vision", [])[1] == "exact_title"
    hit = index.find("CSU101: An Educational Dataset for Vision", ["Ethan Seefried", "Jeremy Lewis", "Cy Three"])
    assert hit is not None and hit[0].ref == "or:1" and hit[1].startswith("retitled_same_authors")
    # Same authors, unrelated title -> a different paper.
    assert index.find("Quantum Widgets for Underwater Robots", ["Ethan Seefried", "Jeremy Lewis", "Cy Three"]) is None
    # Similar title, different authors -> a different paper.
    assert index.find("CSU101: An Educational Dataset for Vision", ["Someone Else"]) is None


def test_resolve_menu_url_from_config(tmp_path: Path) -> None:
    ws = tmp_path / "conf"
    ws.mkdir()
    (ws / "config.yaml").write_text(
        'conference:\n  name: "Conf 2099"\nurls:\n  openaccess: "https://openaccess.thecvf.com/CONF2099"\n',
        encoding="utf-8",
    )
    assert cvf.resolve_menu_url(ws) == MENU_URL
    assert cvf.resolve_menu_url(ws, "https://openaccess.thecvf.com/Other_workshops/menu").endswith("Other_workshops/menu")
    with pytest.raises(ValueError):
        cvf.resolve_menu_url(ws, "https://example.com/menu")
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "config.yaml").write_text('conference:\n  name: "Bare 2099"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        cvf.resolve_menu_url(bare)


def _workspace_with_openreview_paper(tmp_path: Path) -> Path:
    ws = tmp_path / "conf"
    venue = "org/CONF/2099/Workshop/EarthVision"
    receipts = ws / "storage/workshops/papers" / slugify_venueid(venue) / "receipts"
    receipts.mkdir(parents=True)
    (receipts / "n1.json").write_text(
        json.dumps({"status": "succeeded", "title": "Beta", "note_id": "n1", "authors": ["Cy Three"]}),
        encoding="utf-8",
    )
    (ws / "storage/workshops/download_receipt.json").write_text(
        json.dumps({"output_dir": "storage/workshops/papers", "per_workshop": [{"venueid": venue, "status": "succeeded"}]}),
        encoding="utf-8",
    )
    (ws / "storage/workshops/inventory.json").write_text(
        json.dumps({"workshops": [{"venueid": venue, "pdf_availability_class": "openreview_with_pdf"}]}),
        encoding="utf-8",
    )
    return ws


def _fake_site(pages: dict[str, bytes], calls: list[str]):
    def fetch(url: str) -> tuple[int, str, bytes]:
        calls.append(url)
        if url not in pages:
            import urllib.error

            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return 200, "text/html", pages[url]

    return fetch


def test_run_downloads_dedupes_writes_compatible_receipts_and_resumes(tmp_path: Path) -> None:
    ws = _workspace_with_openreview_paper(tmp_path)
    ev_url = "https://openaccess.thecvf.com/CONF2099_workshops/EarthVision"
    listing = _listing(
        _paper_block("A_Paper", "Alpha", ["Ann One"]),
        _paper_block("B_Paper", "Beta", ["Cy Three"]),  # already on OpenReview
    ).encode()
    menu = b'<dd><a href="/CONF2099_workshops/EarthVision">EarthVision</a></dd>'
    pdf_a = "https://openaccess.thecvf.com/content/CONF2099W/X/papers/A_Paper.pdf"
    pages = {MENU_URL: menu, ev_url: listing, pdf_a: _pdf_bytes("a")}
    calls: list[str] = []
    options = cvf.RunOptions(workspace=ws, menu_url=MENU_URL, rate_limit_rps=100.0, workers=2)

    dry = cvf.run(cvf.RunOptions(**{**options.__dict__, "dry_run": True}), fetch=_fake_site(pages, calls))
    assert dry["status"] == "dry_run" and dry["totals"]["planned"] == 1
    assert pdf_a not in calls

    receipt = cvf.run(options, fetch=_fake_site(pages, calls))
    assert receipt["status"] == "succeeded"
    assert receipt["totals"] == {
        "listed": 2, "duplicates_of_openreview": 1, "planned": 1, "attempted": 1,
        "succeeded": 1, "unreachable": 0, "resume_skipped": 0, "available": 1,
    }
    entry = receipt["per_workshop"][0]
    assert entry["openreview_venueid"] == "org/CONF/2099/Workshop/EarthVision"
    assert entry["duplicates_of_openreview"][0]["duplicate_of"] == "org/CONF/2099/Workshop/EarthVision:n1"
    # Same layout/schema the corpus builder reads for OpenReview workshop papers.
    assert receipt["output_dir"] == "storage/workshops/papers"
    workshop_dir = ws / receipt["output_dir"] / slugify_venueid(entry["venueid"])
    paper_receipts = list((workshop_dir / "receipts").glob("*.json"))
    assert len(paper_receipts) == 1
    paper = json.loads(paper_receipts[0].read_text(encoding="utf-8"))
    assert paper["receipt_type"] == "conference_pipeline.workshop_paper_receipt"
    assert paper["status"] == "succeeded" and paper["source"] == "cvf_openaccess"
    assert (workshop_dir / f"{paper['note_id']}.pdf").read_bytes() == _pdf_bytes("a")
    assert paper["byte_count"] == len(_pdf_bytes("a")) and len(paper["sha256"]) == 64

    # A workshop whose papers were all acquired from OpenReview has no receipts
    # directory, so it must not claim ``succeeded`` (corpus builders require one).
    only_dup = cvf.run(
        options, fetch=_fake_site({**pages, ev_url: _listing(_paper_block("B_Paper", "Beta", ["Cy Three"])).encode()}, [])
    )
    assert only_dup["status"] == "succeeded"
    assert only_dup["per_workshop"][0]["status"] == "no_new_papers"

    calls.clear()
    again = cvf.run(options, fetch=_fake_site(pages, calls))
    assert again["totals"]["resume_skipped"] == 1 and again["totals"]["attempted"] == 0
    assert pdf_a not in calls


def test_run_marks_non_pdf_and_missing_listing(tmp_path: Path) -> None:
    ws = _workspace_with_openreview_paper(tmp_path)
    menu = (
        b'<dd><a href="/CONF2099_workshops/EarthVision">EV</a></dd>'
        b'<dd><a href="/CONF2099_workshops/Gone">Gone</a></dd>'
    )
    ev_url = "https://openaccess.thecvf.com/CONF2099_workshops/EarthVision"
    pdf_a = "https://openaccess.thecvf.com/content/CONF2099W/X/papers/A_Paper.pdf"
    pages = {
        MENU_URL: menu,
        ev_url: _listing(_paper_block("A_Paper", "Alpha", ["Ann One"])).encode(),
        pdf_a: b"<html>login</html>",
    }
    receipt = cvf.run(
        cvf.RunOptions(workspace=ws, menu_url=MENU_URL, rate_limit_rps=100.0), fetch=_fake_site(pages, [])
    )
    assert receipt["status"] == "partial"
    by_name = {e["cvf_short_name"]: e for e in receipt["per_workshop"]}
    assert by_name["Gone"]["status"] == "blocked"
    assert by_name["EarthVision"]["unreachable"] == 1


def test_run_blocks_when_menu_unreachable(tmp_path: Path) -> None:
    ws = _workspace_with_openreview_paper(tmp_path)
    receipt = cvf.run(cvf.RunOptions(workspace=ws, menu_url=MENU_URL, rate_limit_rps=100.0), fetch=_fake_site({}, []))
    assert receipt["status"] == "blocked"
