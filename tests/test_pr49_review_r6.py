"""v6.0 Phase 7.1, PR 49 review pr-49-r6: one regression test per finding (F1-F3)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tests.test_release_cli import SLUG, _no_secrets, cli, w0, w4  # noqa: F401
from tests.test_release_reader_families import _ascii_pdf, _pdf_parts

MARKER = "secret-top"
ENV = {"CONFERENCE_PIPELINE_PRIVATE_MARKERS": MARKER}
HOME = "/" + "home/private-review-user/internal-note"
PAGE = b"BT /F1 10 Tf 10 100 Td (hello) Tj ET"


def _zipped(data: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("paper.pdf", data)
    return out.getvalue()


def _nul_hex_pdf() -> bytes:
    # A compressed ActualText hex string with a NUL (PDF whitespace) between its digits.
    return _pdf_parts(b"/Artifact << /ActualText <736563726574\x002d746f70> >> BDC EMC " + PAGE)


def _utf16_xmp_pdf(body: str, encoding: str) -> bytes:
    # Compressed XMP metadata as UTF-16 without a byte-order mark.
    return _pdf_parts(PAGE, xmp=f'<?xml version="1.0"?><private>{body}</private>'.encode(encoding))


def _scan(data: bytes, env: dict[str, str]) -> bool:
    from conference_pipeline._security.private_markers import (
        builtin_rules,
        extra_rules_from_env,
        has_publishing_marker_bytes,
    )

    # The gates' own rule set: an empty rules argument would switch the built-in rules off.
    return has_publishing_marker_bytes(data, rules=builtin_rules() + extra_rules_from_env(env))


def _hub_flags(tmp_path: Path, monkeypatch, data: bytes, env: dict[str, str]) -> bool:
    from conference_pipeline.publishing.release_hub import _marker_findings
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "paper.pdf").write_bytes(data)
    return bool([f for f in _marker_findings(hub, env) if "paper.pdf" in f])


def _release_flags(w4, data: bytes, env: dict[str, str]) -> bool:
    (w4.bundle / "decks" / "01_sample" / "deck.pdf").write_bytes(data)
    cli().release_check(SLUG, env=env, secret_scanner=_no_secrets, **w4.kw())
    return any("private_marker" in line and "deck.pdf" in line for line in w4.out)


# --- F1: a NUL inside a PDF hex string ---------------------------------------------------


@pytest.mark.parametrize("wrap", [False, True])
def test_f1_extra_marker_in_a_hex_string_with_a_nul(wrap: bool) -> None:
    data = _zipped(_nul_hex_pdf()) if wrap else _nul_hex_pdf()
    assert _scan(data, ENV) is True
    assert _scan(data, {}) is False  # control: no marker configured


def test_f1_hub_and_release_check_refuse_it(tmp_path: Path, monkeypatch, w4) -> None:
    assert _hub_flags(tmp_path, monkeypatch, _nul_hex_pdf(), ENV)
    assert _release_flags(w4, _nul_hex_pdf(), ENV), w4.text()


# --- F2: BOM-less UTF-16 in a decoded PDF stream -----------------------------------------


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_f2_utf16_stream_meets_extra_and_builtin_rules(encoding: str, wrap: bool) -> None:
    def pdf(body: str) -> bytes:
        data = _utf16_xmp_pdf(body, encoding)
        return _zipped(data) if wrap else data

    assert _scan(pdf(MARKER), ENV) is True
    assert _scan(pdf(HOME), {}) is True  # a built-in rule: a home path
    assert _scan(pdf("clean words"), ENV) is False  # control


def test_f2_hub_and_release_check_refuse_a_utf16_home_path(tmp_path: Path, monkeypatch, w4) -> None:
    data = _utf16_xmp_pdf(HOME, "utf-16-le")
    assert _hub_flags(tmp_path, monkeypatch, data, {})
    assert _release_flags(w4, data, {}), w4.text()


# --- F3: .env in quotes or parentheses in a PDF comment ----------------------------------


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("comment", [b'% ".env"\n', b"% (.env)\n"])
def test_f3_quoted_env_file_in_a_pdf_comment(comment: bytes, wrap: bool) -> None:
    data = _ascii_pdf("hello") + comment
    assert _scan(_zipped(data) if wrap else data, {}) is True
    # Control: a word holding .env is not the env file.
    assert _scan(_ascii_pdf("hello") + b"% changes.environmental\n", {}) is False


def test_f3_hub_and_release_check_refuse_it(tmp_path: Path, monkeypatch, w4) -> None:
    data = _ascii_pdf("hello") + b'% ".env"\n'
    assert _hub_flags(tmp_path, monkeypatch, data, {})
    assert _release_flags(w4, data, {}), w4.text()
