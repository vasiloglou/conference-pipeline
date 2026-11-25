"""The public-site corpus, cases C1-C8 (v6.0-Public-Release Phase 4).

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


def site():
    return importlib.import_module("conference_pipeline.publishing.static_site")


def checks():
    return importlib.import_module("conference_pipeline.publishing.site_checks")


def build(bundle: Path, out: Path):
    module = site()
    return module.build_public_site(module.PublicSiteConfig(bundle_root=bundle, output_dir=out, receipt_path=out.parent / "site-receipt.json"))


def soup(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")


@pytest.fixture
def alpha_site(tmp_path: Path) -> Path:
    out = tmp_path / "alpha-site"
    build(make_alpha(tmp_path), out)
    return out


@pytest.fixture
def beta_site(tmp_path: Path) -> Path:
    out = tmp_path / "beta-site"
    build(make_beta(tmp_path), out)
    return out


def test_c1_a_workspace_root_is_refused(tmp_path: Path) -> None:
    make_alpha(tmp_path)
    out = tmp_path / "c1-site"
    with pytest.raises(Exception):
        build(tmp_path / "conferences" / "alphaconf", out)
    assert CASES["C1"]["build_exits_nonzero"] is True
    assert not out.exists()


def _topic_facts(page: BeautifulSoup) -> dict:
    facts = {"deck_html": False, "deck_pdf": False, "videos": 0, "podcast": False, "narrative": False, "papers": []}
    for a in page.select("[data-item=deck] a[href]"):
        facts["deck_html"] |= a["href"].endswith(".html")
        facts["deck_pdf"] |= a["href"].endswith(".pdf")
    facts["videos"] = len(page.select("video[controls] [src], video[controls][src]"))
    facts["podcast"] = bool(page.select("audio[controls]"))
    facts["narrative"] = bool(page.select("[data-item=narrative] article"))
    facts["papers"] = [li["data-paper-id"] for li in page.select("[data-item=papers] [data-paper-id]")]
    return facts


def test_c2_each_topic_page_shows_what_the_topic_has(alpha_site: Path) -> None:
    e = CASES["C2"]
    for rel in e["topic_pages"]:
        assert (alpha_site / rel).is_file(), rel
    for topic in ("01_reasoning", "02_agents"):
        assert _topic_facts(soup(alpha_site / f"topics/{topic}.html")) == e[topic], topic
    agents = soup(alpha_site / "topics/02_agents.html").get_text(" ")
    assert agents.count("not in this release") >= 4  # pdf, video, podcast, narrative
    first = soup(alpha_site / "topics/01_reasoning.html")
    srcs = [s.get("src") for s in first.select("video[controls] source, video[controls]") if s.get("src")]
    assert srcs and all(src.startswith("https://cdn.example.test/") for src in srcs)
    index = soup(alpha_site / "index.html")
    order = [a["href"] for a in index.select("[data-topic] a[href^='topics/']")]
    assert order == ["topics/01_reasoning.html", "topics/02_agents.html"]


def test_c3_a_paper_with_no_public_copy(alpha_site: Path) -> None:
    page = soup(alpha_site / "papers.html")
    (row,) = page.select("[data-paper-id='site:s1']")
    assert "no public copy known" in row.get_text(" ")
    assert row.select("a[href]") == []


def test_c4_every_link_resolves_and_nothing_private_is_copied(alpha_site: Path) -> None:
    result = checks().check_links(alpha_site)
    assert result.exit_code == CASES["C4"]["check_links_exit"], result.findings
    files = [p.relative_to(alpha_site).as_posix() for p in alpha_site.rglob("*") if p.is_file()]
    assert not any(f.startswith("public-repo/") or "/.git/" in f or f.startswith(".git/") for f in files)
    for page in alpha_site.rglob("*.html"):
        if page.name == "data.html" or "decks/" in page.relative_to(alpha_site).as_posix():
            continue
        hrefs = [a.get("href", "") for a in soup(page).select("a[href]")]
        assert not any(h.endswith(("RELEASE.json", "CHECKSUMS.json", "presentation_manifest.json")) for h in hrefs), page


def test_c5_full_text_ranks_the_narrative_first(alpha_site: Path) -> None:
    ranked = checks().search_text(alpha_site, "quokkalike")
    assert ranked and ranked[0]["source_path"] == CASES["C5"]["first_result_source"]


def _modes(page: Path) -> list[str]:
    return [b["data-mode"] for b in soup(page).select("[data-mode]")]


def test_c6_semantic_is_shown_when_the_bundle_has_vectors(alpha_site: Path) -> None:
    e = CASES["C6"]
    assert _modes(alpha_site / "search.html") == e["modes_shown"]
    config = json.loads((alpha_site / "search-config.json").read_text(encoding="utf-8"))
    assert {"model": config["semantic"]["model"], "dim": config["semantic"]["dim"]} == e["search_config"]


def test_c7_semantic_is_hidden_when_it_cannot_work(beta_site: Path) -> None:
    assert _modes(beta_site / "search.html") == CASES["C7"]["modes_shown"]
    text = soup(beta_site / "search.html").get_text(" ").lower()
    assert "full-text search only" in text
    assert "semantic" not in "".join(_modes(beta_site / "search.html"))


def test_c8_a_tampered_bundle_file_stops_the_build(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    path = bundle / "narratives" / "01_reasoning.md"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(site().StaticSiteError) as raised:
        build(bundle, tmp_path / "c8-site")
    assert CASES["C8"]["error"] in str(raised.value)


# --- implementation successors (spec section 12, found by the real-bundle run T10) ------


def _refinish(root: Path) -> None:
    from tests.public_site_fixtures import _finish

    _finish(root)


def test_if7_a_refused_build_leaves_no_site_behind(tmp_path: Path) -> None:
    """A copied file with a home-directory path stops the build, and the half-written site,
    which already holds that file, is removed (IF7)."""
    bundle = make_alpha(tmp_path)
    planted = "/" + "Users" + "/qzxplanted/notes"  # split so the tracked-tree scan stays clean
    (bundle / "decks" / "markdown" / "01_reasoning" / "slides.md").write_text(f"# Slides\n\nsee {planted}\n", encoding="utf-8")
    _refinish(bundle)
    out = tmp_path / "if7-site"
    with pytest.raises(site().StaticSiteError, match="private marker"):
        build(bundle, out)
    assert not out.exists()


def test_if8_a_release_without_index_or_catalog_says_so(tmp_path: Path) -> None:
    """The real v1.0.0 bundle has neither index/ nor catalog/: the search and papers pages
    say "not in this release" instead of offering a search over nothing or "0 papers" (IF8)."""
    import shutil

    bundle = make_beta(tmp_path)
    shutil.rmtree(bundle / "index")
    shutil.rmtree(bundle / "catalog")
    _refinish(bundle)
    out = tmp_path / "if8-site"
    build(bundle, out)
    search = soup(out / "search.html")
    assert "Search: not in this release." in search.get_text(" ")
    assert search.find("form") is None and search.find("script") is None
    assert "Papers: not in this release." in soup(out / "papers.html").get_text(" ")
    manifest = json.loads((out / "site_manifest.json").read_text(encoding="utf-8"))
    assert manifest["search_modes"] == []
    assert manifest["semantic_search"]["status"] == "not_in_this_release"
    assert checks().check_links(out).exit_code == 0
    assert checks().lint_accessibility(out).exit_code == 0


def test_if8_an_empty_catalog_still_counts_zero(tmp_path: Path) -> None:
    """A catalog that is present but empty is a real count, not an absence."""
    bundle = make_beta(tmp_path)
    (bundle / "catalog" / "papers.jsonl").write_text("", encoding="utf-8")
    (bundle / "catalog" / "extra_papers.jsonl").unlink()  # else re-checksumming admits it
    _refinish(bundle)
    out = tmp_path / "if8b-site"
    build(bundle, out)
    assert "0 papers." in soup(out / "papers.html").get_text(" ")
