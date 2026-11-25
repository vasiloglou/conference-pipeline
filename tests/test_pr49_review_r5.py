"""v6.0 Phase 7.1, PR 49 review pr-49-r5: one regression test per finding (F1-F2)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tests.test_release_cli import SLUG, _no_secrets, cli, w0, w4  # noqa: F401
from tests.test_release_reader_families import _ascii_pdf, _pdf_parts


def _env(marker: str) -> dict[str, str]:
    return {"CONFERENCE_PIPELINE_PRIVATE_MARKERS": marker}


def _actual_text_pdf(hexed: bytes) -> bytes:
    # A Flate-compressed page whose marked-content ActualText is a hex string.
    return _pdf_parts(b"/Artifact << /ActualText <" + hexed + b"> >> BDC EMC BT /F1 10 Tf 10 100 Td (hello) Tj ET")


def _zipped(data: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("paper.pdf", data)
    return out.getvalue()


def _env_comment_pdf() -> bytes:
    # A clean ASCII PDF with a comment naming the env file, appended after %%EOF.
    return _ascii_pdf("hello") + b"% .env\n"


HEX_CASES = {
    "odd_length": (b"7365637265742d746f7", "secret-top"),
    "short": (b"78797a", "xyz"),
    "utf16_bom": (b"feff" + "secret-top".encode("utf-16-be").hex().encode(), "secret-top"),
}


# --- F1: extra markers in odd, short and UTF-16 PDF hex strings -------------------------


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("case", sorted(HEX_CASES))
def test_f1_extra_marker_in_a_pdf_hex_string(case: str, wrap: bool) -> None:
    from conference_pipeline._security.private_markers import extra_rules_from_env, has_publishing_marker_bytes

    hexed, marker = HEX_CASES[case]
    data = _actual_text_pdf(hexed)
    if wrap:
        data = _zipped(data)
    assert has_publishing_marker_bytes(data, rules=extra_rules_from_env(_env(marker))) is True
    # Control: the same PDF with no marker configured is clean.
    assert has_publishing_marker_bytes(data, rules=extra_rules_from_env({})) is False


def test_f1_hub_refuses_an_odd_hex_marker(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing.release_hub import _marker_findings
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "paper.pdf").write_bytes(_actual_text_pdf(HEX_CASES["odd_length"][0]))
    assert [f for f in _marker_findings(hub, _env("secret-top")) if "paper.pdf" in f]
    assert not [f for f in _marker_findings(hub, {}) if "paper.pdf" in f]


def test_f1_release_check_refuses_an_odd_hex_marker(w4) -> None:
    (w4.bundle / "decks" / "01_sample" / "deck.pdf").write_bytes(_actual_text_pdf(HEX_CASES["odd_length"][0]))
    cli().release_check(SLUG, env=_env("secret-top"), secret_scanner=_no_secrets, **w4.kw())
    assert any("private_marker" in line and "deck.pdf" in line for line in w4.out), w4.text()


# --- F2: the env file named in a PDF comment ---------------------------------------------


@pytest.mark.parametrize("wrap", [False, True])
def test_f2_env_file_in_a_pdf_comment(wrap: bool) -> None:
    from conference_pipeline._security.private_markers import has_publishing_marker_bytes

    data = _env_comment_pdf()
    assert has_publishing_marker_bytes(_zipped(data) if wrap else data) is True
    # Controls: a word holding .env, and the same PDF without the comment, are clean.
    assert has_publishing_marker_bytes(_ascii_pdf("changes.environmental changes")) is False
    assert has_publishing_marker_bytes(_ascii_pdf("hello")) is False


def test_f2_hub_flags_the_env_file_in_a_pdf_comment(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing.release_hub import _marker_findings
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "paper.pdf").write_bytes(_env_comment_pdf())
    assert [f for f in _marker_findings(hub, {}) if "paper.pdf" in f]


def test_f2_release_check_flags_the_env_file_in_a_pdf_comment(w4) -> None:
    (w4.bundle / "decks" / "01_sample" / "deck.pdf").write_bytes(_env_comment_pdf())
    cli().release_check(SLUG, env={}, secret_scanner=_no_secrets, **w4.kw())
    assert any("private_marker" in line and "deck.pdf" in line for line in w4.out), w4.text()
