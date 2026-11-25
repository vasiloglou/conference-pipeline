"""Supplementary proceedings tracks (e.g. CVF Findings) as main-conference papers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.acquisition import proceedings_paper_sources as sources

LISTING = """
<dl>
<dt class="ptitle"><br><a href="/content/C/html/A_Paper_One_paper.html">Paper One: Findings</a></dt>
<dd><form><input type="hidden" name="query_author" value="Ada Lovelace"></form>
[<a href="/content/C/papers/A_Paper_One_paper.pdf">pdf</a>]</dd>
<dt class="ptitle"><br><a href="/content/C/html/B_Main_Dup_paper.html">Already In Main Track</a></dt>
<dd>[<a href="/content/C/papers/B_Main_Dup_paper.pdf">pdf</a>]</dd>
<dt class="ptitle"><br><a href="/content/C/html/C_Broken_paper.html">Broken Paper</a></dt>
<dd>[<a href="/content/C/papers/C_Broken_paper.pdf">pdf</a>]</dd>
</dl>
"""


def _fetch(url: str) -> tuple[int, str, bytes]:
    if url.endswith("findings?day=all"):
        return 200, "text/html", LISTING.encode()
    if "C_Broken" in url:
        return 200, "text/html", b"<html>not a pdf</html>"
    return 200, "application/pdf", b"%PDF-1.7 " + url.encode()


def _download(workspace: Path, **kwargs):  # noqa: ANN003, ANN202
    return sources.download_cvf_proceedings_listing(
        workspace=workspace,
        source_id="cvf_findings",
        listing_url="https://openaccess.thecvf.com/X_findings?day=all",
        track="findings",
        venue="X Findings",
        known_titles=["Already in main track"],
        rate_limit_rps=100.0,
        fetch=_fetch,
        **kwargs,
    )


def test_download_skips_known_titles_and_records_failures(tmp_path: Path) -> None:
    receipt = _download(tmp_path)
    assert receipt["status"] == "partial"
    assert receipt["counts"] == {"succeeded": 1, "duplicate_of_known_title": 1, "unreachable": 1}
    ok = next(entry for entry in receipt["papers"] if entry["status"] == "succeeded")
    assert ok["authors"] == ["Ada Lovelace"] and ok["title"] == "Paper One: Findings"
    assert (tmp_path / ok["pdf_path"]).read_bytes().startswith(b"%PDF-")
    assert json.loads(sources.receipt_path(tmp_path, "cvf_findings").read_text())["track"] == "findings"


def test_resume_keeps_verified_downloads_and_registry_pins_digest(tmp_path: Path) -> None:
    _download(tmp_path)
    calls: list[str] = []

    def counting(url: str):  # noqa: ANN202
        calls.append(url)
        return _fetch(url)

    sources.download_cvf_proceedings_listing(
        workspace=tmp_path,
        source_id="cvf_findings",
        listing_url="https://openaccess.thecvf.com/X_findings?day=all",
        track="findings",
        venue="X Findings",
        known_titles=["Already in main track"],
        rate_limit_rps=100.0,
        fetch=counting,
    )
    assert not any("A_Paper_One" in url for url in calls), "a verified download is not fetched again"

    sources.register_source(workspace=tmp_path, source_id="cvf_findings")
    loaded, blockers = sources.load_registered_sources(tmp_path)
    assert blockers == [] and loaded[0]["track"] == "findings"
    receipt = sources.receipt_path(tmp_path, "cvf_findings")
    receipt.write_text(receipt.read_text() + " ", encoding="utf-8")
    _loaded, blockers = sources.load_registered_sources(tmp_path)
    assert [blocker["code"] for blocker in blockers] == ["proceedings_receipt_digest_mismatch"]


def test_invalid_source_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sources.download_cvf_proceedings_listing(
            workspace=tmp_path, source_id="../escape", listing_url="x", track="t", venue="v", fetch=_fetch
        )
