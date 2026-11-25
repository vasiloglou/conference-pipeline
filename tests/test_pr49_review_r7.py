"""v6.0 Phase 7.1, PR 49 review pr-49-r7: one regression test per finding (F1-F3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_pr49_review_r6 import ENV, HOME, MARKER, _hub_flags, _release_flags, _scan, _utf16_xmp_pdf, _zipped
from tests.test_release_cli import _no_secrets, cli, w0, w4  # noqa: F401
from tests.test_release_reader_families import _ascii_pdf

CYRILLIC = "Пример текста для проверки. " * 100
CJK = "检查文本示例。" * 300


def _pdf(objs: list[bytes], prefix: bytes = b"") -> bytes:
    out, offsets = b"%PDF-1.4\n" + prefix, []
    for number, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def _image_pdf(image: bytes, prefix: bytes = b"") -> bytes:
    # One page drawing a DCTDecode image: its encoded bytes are opaque.
    content = b"q 10 0 0 10 0 0 cm /Im1 Do Q"
    return _pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R"
        b" /Resources << /XObject << /Im1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8"
        b" /Filter /DCTDecode /Length %d >>\nstream\n" % len(image) + image + b"\nendstream",
    ], prefix)


# --- F1: UTF-16 text the byte-order heuristic does not recognise --------------------------


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("prefix", [CYRILLIC, CJK], ids=["cyrillic", "cjk"])
def test_f1_utf16_stream_of_non_latin_text_meets_every_rule(prefix: str, encoding: str, wrap: bool) -> None:
    def pdf(body: str) -> bytes:
        data = _utf16_xmp_pdf(prefix + body, encoding)
        return _zipped(data) if wrap else data

    assert _scan(pdf(MARKER), ENV) is True
    assert _scan(pdf(HOME), {}) is True
    assert _scan(pdf(".env"), {}) is True
    assert _scan(pdf(" clean words"), ENV) is False  # control


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("prefix", [CYRILLIC, CJK], ids=["cyrillic", "cjk"])
def test_f1_a_utf16_text_file_of_non_latin_text_meets_every_rule(prefix: str, encoding: str) -> None:
    assert _scan((prefix + HOME).encode(encoding), {}) is True
    assert _scan((prefix + MARKER).encode(encoding), ENV) is True
    assert _scan((prefix + " clean words").encode(encoding), ENV) is False  # control


def test_f1_hub_and_release_check_refuse_it(tmp_path: Path, monkeypatch, w4) -> None:
    data = _utf16_xmp_pdf(CYRILLIC + HOME, "utf-16-le")
    assert _hub_flags(tmp_path, monkeypatch, data, {})
    assert _release_flags(w4, data, {}), w4.text()


# --- F2: a comment that looks like stream data ------------------------------------------

FAKE_STREAM = b"% \x00\n% stream\n% .env\n% endstream\n"


@pytest.mark.parametrize("wrap", [False, True])
def test_f2_a_comment_framed_as_a_stream_is_still_read(wrap: bool) -> None:
    data = _ascii_pdf("hello") + FAKE_STREAM
    assert _scan(_zipped(data) if wrap else data, {}) is True
    # Control: the same frame around a word holding .env.
    clean = _ascii_pdf("hello") + FAKE_STREAM.replace(b".env", b"changes.environmental")
    assert _scan(clean, {}) is False


def test_f2_non_stream_text_meets_every_rule() -> None:
    # A comment outside every stream is text: a home path there is refused too.
    assert _scan(_ascii_pdf("hello") + b"% \x00\n% stream\n% " + HOME.encode() + b"\n% endstream\n", {}) is True


def test_f2_hub_and_release_check_refuse_it(tmp_path: Path, monkeypatch, w4) -> None:
    data = _ascii_pdf("hello") + FAKE_STREAM
    assert _hub_flags(tmp_path, monkeypatch, data, {})
    assert _release_flags(w4, data, {}), w4.text()


# --- F3: opaque image bytes meet the separator rule in every gate ------------------------

# A JPEG whose APP15 segment holds .env between two non-ASCII bytes, after a long comment
# that keeps the first 8 KiB free of NUL bytes.
JPEG = b"\xff\xd8\xff\xef\x00\x08\x81.env\x82\xff\xd9"
LONG_COMMENT = b"%" + b"x" * 8200 + b"\n"


def test_f3_opaque_image_bytes_pass_the_byte_check() -> None:
    data = _image_pdf(JPEG, LONG_COMMENT)
    assert b"\0" not in data[:8192]
    assert _scan(data, {}) is False
    # Control: after a path separator, .env is the env file even in opaque bytes.
    assert _scan(_image_pdf(JPEG.replace(b"\x81.env", b"/.env"), LONG_COMMENT), {}) is True


def test_f3_hub_and_release_check_give_the_byte_check_verdict(tmp_path: Path, monkeypatch, w4) -> None:
    data = _image_pdf(JPEG, LONG_COMMENT)
    assert not _hub_flags(tmp_path, monkeypatch, data, {})
    assert not _release_flags(w4, data, {}), w4.text()


def test_f3_hub_and_release_check_still_refuse_a_separated_env(tmp_path: Path, monkeypatch, w4) -> None:
    data = _image_pdf(JPEG.replace(b"\x81.env", b"/.env"), LONG_COMMENT)
    assert _hub_flags(tmp_path, monkeypatch, data, {})
    assert _release_flags(w4, data, {}), w4.text()


# --- the zip checker reads member names, which the raw text scan used to --------------


def test_zip_member_name_meets_every_rule() -> None:
    import io
    import zipfile

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(HOME.lstrip("/") + "/notes.txt", "clean words")
        archive.writestr("deploy/.env", "clean words")
    assert _scan(out.getvalue(), {}) is True
    clean = io.BytesIO()
    with zipfile.ZipFile(clean, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("changes.environmental/notes.txt", "clean words")
    assert _scan(clean.getvalue(), {}) is False
