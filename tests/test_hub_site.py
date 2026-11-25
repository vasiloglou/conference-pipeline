"""The hub corpus, cases C9-C12 (v6.0-Public-Release Phase 4).

Stories: tests/fixtures/public_site/README.md; answers: EXPECTED.json.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from tests.public_site_fixtures import make_alpha, make_beta

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((REPO_ROOT / "tests" / "fixtures" / "public_site" / "EXPECTED.json").read_text(encoding="utf-8"))["cases"]


def hub():
    return importlib.import_module("conference_pipeline.publishing.hub_site")


@pytest.fixture
def hub_dir(tmp_path: Path) -> Path:
    out = tmp_path / "hub"
    hub().build_hub_site([make_alpha(tmp_path), make_beta(tmp_path)], out, hub_title="Fixture hub")
    return out


def _manifest(hub_dir: Path) -> dict:
    return json.loads((hub_dir / "hub_manifest.json").read_text(encoding="utf-8"))


def _rows(hub_dir: Path) -> list[dict]:
    return json.loads((hub_dir / "catalog.json").read_text(encoding="utf-8"))


def test_c9_the_hub_holds_every_public_row(hub_dir: Path) -> None:
    e = CASES["C9"]
    manifest = _manifest(hub_dir)
    assert manifest["rows_total"] == e["rows_total"] == len(_rows(hub_dir))
    assert {rid: m["rows_read"] for rid, m in manifest["releases"].items()} == e["rows_by_release"]


def test_c10_the_acceptance_query_returns_both_conferences(hub_dir: Path) -> None:
    from conference_pipeline.publishing.site_checks import search_text

    e = CASES["C10"]
    results = search_text(hub_dir, "diffusion")
    conferences = {r["conference"] for r in results}
    assert set(e["results_include_conferences"]) <= conferences
    for r in results:
        assert r["release_id"] and r["conference"]
        assert r["id"] == f"{r['release_id']}/{r['paper_id']}"


def test_c11_excluded_and_unchecksummed_rows_never_appear(hub_dir: Path) -> None:
    e = CASES["C11"]
    ids = {r["id"] for r in _rows(hub_dir)}
    everything = "".join(p.read_text(encoding="utf-8") for p in hub_dir.rglob("*") if p.is_file() and p.name != "hub_manifest.json")
    for absent in e["absent"]:
        assert absent not in ids
        assert absent.split("/", 1)[1] not in everything
    beta = _manifest(hub_dir)["releases"]["betaconf-v1.0.0"]
    assert {k: beta[k] for k in e["hub_manifest"]["betaconf-v1.0.0"]} == e["hub_manifest"]["betaconf-v1.0.0"]


def test_c12_mixed_capabilities_mean_full_text_only(hub_dir: Path) -> None:
    modes = [b["data-mode"] for b in BeautifulSoup((hub_dir / "index.html").read_text(encoding="utf-8"), "html.parser").select("[data-mode]")]
    assert modes == CASES["C12"]["modes_shown"]


def test_b37_a_title_several_rows_share_is_named_apart(tmp_path: Path) -> None:
    """A paper title shared by several hub rows (two workshops of one release, or two
    conferences) is named apart by its conference, track and id, as a bundle's own catalog
    names it apart (6b: 1,341 duplicate-link findings on the five-conference hub)."""
    from tests.public_site_fixtures import _finish
    from conference_pipeline.publishing.site_checks import lint_accessibility

    alpha, beta = make_alpha(tmp_path), make_beta(tmp_path)
    shared = "Scaling test time search"  # alpha's openreview:A1
    extra = [
        {"paper_id": "openreview:B4", "title": shared, "authors": [], "track": "workshop", "venue": "Beta 2025", "workshop": "WS", "official_url": "https://openreview.net/forum?id=B4", "topics": []},
        {"paper_id": "openreview:B5", "title": shared, "authors": [], "track": "main", "venue": "Beta 2025", "workshop": None, "official_url": "https://openreview.net/forum?id=B5", "topics": []},
    ]
    with (beta / "catalog" / "papers.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in extra))
    _finish(beta, not_in_checksums=("catalog/extra_papers.jsonl",))
    out = tmp_path / "hub"
    hub().build_hub_site([alpha, beta], out, hub_title="Fixture hub")

    findings = [f for f in lint_accessibility(out).findings if "share the text" in f]
    assert findings == []
    doc = BeautifulSoup((out / "catalog.html").read_text(encoding="utf-8"), "html.parser")
    labels = sorted(a.get("aria-label") for a in doc.select("ul.papers a") if a.get_text(strip=True) == shared)
    assert len(labels) == 3 and all(labels) and len(set(labels)) == 3
    assert any("openreview:A1" in label for label in labels)
    assert any("openreview:B4" in label and "workshop" in label for label in labels)
    # A title only one row carries keeps its plain text as its name.
    assert all(a.get("aria-label") is None for a in doc.select("ul.papers a") if a.get_text(strip=True) == "Vision transformers revisited")


def test_r1_f7_two_releases_of_one_conference_are_named_apart(tmp_path: Path) -> None:
    """A repeated title in two releases of one conference (same paper, track and id) is named
    apart by its release id too (review phase-6b-r1 F7)."""
    from tests.public_site_fixtures import _finish
    from conference_pipeline.publishing.site_checks import lint_accessibility

    alpha = make_alpha(tmp_path / "a")
    again = make_alpha(tmp_path / "b")
    release = json.loads((again / "RELEASE.json").read_text(encoding="utf-8"))
    old_id = release["bundle_id"]
    new_id = old_id.replace("v1.0.0", "v1.1.0")
    assert new_id != old_id
    for path in again.rglob("*.json*"):
        path.write_text(path.read_text(encoding="utf-8").replace(old_id, new_id), encoding="utf-8")
    papers = again / "catalog" / "papers.jsonl"
    papers.write_text(papers.read_text(encoding="utf-8").replace("forum?id=A1", "forum?id=A1v2"), encoding="utf-8")
    _finish(again)
    out = tmp_path / "hub"
    hub().build_hub_site([alpha, again], out, hub_title="Fixture hub")
    assert [f for f in lint_accessibility(out).findings if "share the text" in f] == []


def test_r3_f8_a_hub_rebuilt_later_is_the_same_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = [make_alpha(tmp_path), make_beta(tmp_path)]
    builds = []
    for i, clock in enumerate(["2026-01-01T00:00:00Z", "2027-06-30T12:34:56Z"]):
        monkeypatch.setattr(hub(), "utc_now", lambda clock=clock: clock, raising=False)
        out = tmp_path / f"hub{i}"
        hub().build_hub_site(roots, out, hub_title="Fixture hub")
        builds.append({p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()})
    assert builds[0] == builds[1]


def test_t15_the_hub_and_a_site_render_one_title_the_same_way(tmp_path: Path) -> None:
    """v6.0 Phase 7.1 (D7): one helper renders a paper in both lists."""
    from conference_pipeline.publishing import site_templates as tpl
    from conference_pipeline.publishing.static_site import _paper_item

    row = {"paper_id": "p1", "title": "A  title", "official_url": "https://example.test/p1", "track": "main"}
    site = _paper_item(row, repeated=True)
    hub_row = {"id": "x-v1/p1", "paper_id": "p1", "title": "A  title", "href": "https://example.test/p1",
               "conference": "X", "release_id": "x-v1", "track": "main"}
    hub_html = hub()._catalog_item(hub_row, repeated=True)
    for html in (site, hub_html):
        link = BeautifulSoup(html, "html.parser").find("a")
        assert link.text == "A  title" and link["href"] == "https://example.test/p1"
        assert link["aria-label"].startswith("A  title (")
    assert tpl.repeated_titles(["A  title", "A title", "B"]) == {"A title"}
    assert "(no public copy known)" in tpl.paper_item(item_id="i", title="t", href=None, meta="", distinguish=[], repeated=False)
