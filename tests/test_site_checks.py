"""Site checks over the fixture site and hub: T4-T6 of the Phase 4 spec, and the sanitizer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline.publishing import hub_site, static_site
from conference_pipeline.publishing.site_checks import (
    check_links,
    lint_accessibility,
    render_mobile,
    sanitize_html,
    search_text,
)
from tests.public_site_fixtures import make_alpha, make_beta

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def built(tmp_path: Path) -> dict[str, Path]:
    alpha, beta = make_alpha(tmp_path), make_beta(tmp_path)
    out = {"alpha": tmp_path / "alpha-site", "beta": tmp_path / "beta-site", "hub": tmp_path / "hub"}
    for name, bundle in (("alpha", alpha), ("beta", beta)):
        static_site.build_public_site(static_site.PublicSiteConfig(bundle, out[name], tmp_path / f"{name}-receipt.json"))
    hub_site.build_hub_site([alpha, beta], out["hub"], hub_title="Fixture hub")
    return out


@pytest.mark.parametrize("which", ["alpha", "beta", "hub"])
def test_t4_built_sites_carry_no_private_marker(built: dict[str, Path], which: str) -> None:
    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "audit_private_markers.py"), "--tree", str(built[which]), "--no-env"],
        capture_output=True, text=True, env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("which", ["alpha", "beta", "hub"])
def test_t5_accessibility_lint_is_clean(built: dict[str, Path], which: str) -> None:
    result = lint_accessibility(built[which])
    assert result.exit_code == 0, result.findings


@pytest.mark.parametrize("which", ["alpha", "beta", "hub"])
def test_links_resolve(built: dict[str, Path], which: str) -> None:
    result = check_links(built[which])
    assert result.exit_code == 0, result.findings


PLANTED = {
    "lang": ('<html lang="en">', "<html>", "has no lang"),
    "h1": ("<h1>", "<h2>", "exactly one <h1>"),
    "main": ('<main id="main">', '<div id="main">', "no <main>"),
    "skip": ('<a class="skip" href="#main">Skip to content</a>', "", "no skip link"),
    "img": ("</main>", '<img src="x.png"></main>', "<img> without alt"),
    "control": ("</main>", '<input name="x"></main>', "without an accessible name"),
    "button": ("</main>", "<button></button></main>", "button without text"),
    "media": ("</main>", '<audio src="x.mp3"></audio></main>', "without controls"),
    "links": ("</main>", '<a href="a.html">Open</a><a href="b.html">Open</a></main>', "share the text"),
    "heading": ("</main>", "<h4>Deep</h4></main>", "heading level skips"),
}


@pytest.mark.parametrize("rule", sorted(PLANTED))
def test_t5_each_rule_fails_on_its_planted_defect(built: dict[str, Path], rule: str) -> None:
    before, after, message = PLANTED[rule]
    page = built["beta"] / "papers.html"
    text = page.read_text(encoding="utf-8")
    assert before in text, rule
    page.write_text(text.replace(before, after, 1), encoding="utf-8")
    result = lint_accessibility(built["beta"])
    assert result.exit_code == 1
    assert any(message in f for f in result.findings), result.findings


def test_links_fail_on_a_missing_file_and_a_bare_http_link(built: dict[str, Path]) -> None:
    page = built["beta"] / "data.html"
    page.write_text(page.read_text(encoding="utf-8").replace("</main>", '<a href="files/nope.pdf">x</a><a href="http://example.test/">y</a></main>'), encoding="utf-8")
    result = check_links(built["beta"])
    assert result.exit_code == 1
    assert any("no such file" in f for f in result.findings) and any("not https" in f for f in result.findings)


@pytest.mark.parametrize("which", ["alpha", "hub"])
def test_t6_pages_fit_a_phone_screen(built: dict[str, Path], which: str) -> None:
    result = render_mobile(built[which], width=375)
    if result.skipped:
        pytest.skip(f"mobile render skipped: {result.skipped}")
    assert result.exit_code == 0, result.findings


def test_t6_a_too_wide_page_fails(built: dict[str, Path]) -> None:
    page = built["beta"] / "papers.html"
    page.write_text(page.read_text(encoding="utf-8").replace("</main>", '<div style="width:2000px">wide</div></main>'), encoding="utf-8")
    result = render_mobile(built["beta"], width=375)
    if result.skipped:
        pytest.skip(f"mobile render skipped: {result.skipped}")
    assert result.exit_code == 1 and any("overflow" in f for f in result.findings)


def test_browser_full_text_ranks_like_the_reference(built: dict[str, Path]) -> None:
    """search.js, run in Chromium over the alpha site, ranks as site_checks.bm25_rank does."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed")
    import http.server
    import socketserver
    import threading

    site = built["alpha"]
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(site), **k)  # noqa: E731
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/search.html")
            page.fill("#q", "quokkalike search")
            page.click("#search-form button[type=submit]")
            page.wait_for_selector("#results li")
            first = page.inner_text("#results li >> nth=0")
            browser.close()
        server.shutdown()
    expected = search_text(site, "quokkalike search")[0]
    assert expected["title"] in first


def test_sanitizer_drops_scripts_handlers_and_unsafe_links() -> None:
    dirty = '<h1>T</h1><script>alert(1)</script><p onclick="x()">ok <a href="javascript:x()">j</a> <a href="https://e.test/">h</a></p>'
    clean = sanitize_html(dirty, heading_offset=2)
    assert "<script" not in clean and "alert" not in clean and "onclick" not in clean
    assert "javascript:" not in clean and 'href="https://e.test/"' in clean
    assert clean.startswith("<h3>T</h3>")
