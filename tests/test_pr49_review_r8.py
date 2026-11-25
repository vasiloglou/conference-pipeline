"""v6.0 Phase 7.1, PR 49 review pr-49-r8: one regression test per finding (F1-F5)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tests.test_pr49_review_r6 import HOME, _hub_flags, _release_flags, _scan, _utf16_xmp_pdf, _zipped
from tests.test_pr49_review_r7 import _image_pdf
from tests.test_release_cli import _no_secrets, cli, w0, w4  # noqa: F401
from tests.test_release_reader_families import _ascii_pdf

ENCODINGS = ["utf-16-le", "utf-16-be"]
CYRILLIC_HOME = "/" + "home/топ"
CYRILLIC_MARKER = "secret-топ"
CYRILLIC_ENV = {"CONFERENCE_PIPELINE_PRIVATE_MARKERS": CYRILLIC_MARKER}


# --- F1: non-ASCII markers in UTF-16 text --------------------------------------------------


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_f1_non_ascii_markers_in_a_utf16_stream(encoding: str, wrap: bool) -> None:
    def pdf(body: str) -> bytes:
        data = _utf16_xmp_pdf(body, encoding)
        return _zipped(data) if wrap else data

    assert _scan(pdf(CYRILLIC_HOME), {}) is True
    assert _scan(pdf(CYRILLIC_MARKER), CYRILLIC_ENV) is True
    assert _scan(pdf("чистый текст"), CYRILLIC_ENV) is False  # control


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("offset", [0, 1])
def test_f1_non_ascii_home_path_in_utf16_bytes_at_either_alignment(encoding: str, offset: int) -> None:
    data = b"\x00" * offset + b"\x89BIN" + ("x " + CYRILLIC_HOME + " y").encode(encoding)
    assert _scan(data, {}) is True


def test_f1_hub_and_release_check_refuse_it(tmp_path: Path, monkeypatch, w4) -> None:
    data = _utf16_xmp_pdf(CYRILLIC_HOME, "utf-16-be")
    assert _hub_flags(tmp_path, monkeypatch, data, {})
    assert _release_flags(w4, data, {}), w4.text()


# --- F2: a comment holding a copy of a stream's bytes ------------------------------------

IMAGE = b"\xff\xd8\xff\xef\x00\x0a\n% .env\n\xff\xd9"


def _copied_stream_pdf(secret: bytes) -> bytes:
    image = IMAGE.replace(b".env", secret)
    return _image_pdf(image) + b"\n% stream\n" + image + b"\n% endstream\n"


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("secret", [b".env", HOME.encode()], ids=["env", "home"])
def test_f2_a_comment_copying_stream_bytes_is_still_read(secret: bytes, wrap: bool) -> None:
    data = _copied_stream_pdf(secret)
    assert _scan(_zipped(data) if wrap else data, {}) is True


def test_f2_the_real_stream_stays_opaque() -> None:
    # Control: the same bytes only inside the image stream are opaque (no separator).
    assert _scan(_image_pdf(IMAGE.replace(b"% .env", b"\x81.env\x82")), {}) is False


def test_f2_hub_and_release_check_refuse_it(tmp_path: Path, monkeypatch, w4) -> None:
    data = _copied_stream_pdf(b".env")
    assert _hub_flags(tmp_path, monkeypatch, data, {})
    assert _release_flags(w4, data, {}), w4.text()


# --- F3: UTF-16 text outside every PDF stream --------------------------------------------


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("secret", [HOME, ".env ", CYRILLIC_HOME], ids=["home", "env", "cyrillic-home"])
def test_f3_utf16_text_in_a_pdf_comment(secret: str, encoding: str, wrap: bool) -> None:
    data = _ascii_pdf("hello") + b"\n% " + secret.encode(encoding) + b"\n"
    assert _scan(_zipped(data) if wrap else data, {}) is True
    clean = _ascii_pdf("hello") + b"\n% " + "changes.environmental".encode(encoding) + b"\n"
    assert _scan(clean, {}) is False  # control


def test_f3_hub_and_release_check_refuse_it(tmp_path: Path, monkeypatch, w4) -> None:
    data = _ascii_pdf("hello") + b"\n% " + HOME.encode("utf-16-le") + b"\n"
    assert _hub_flags(tmp_path, monkeypatch, data, {})
    assert _release_flags(w4, data, {}), w4.text()


# --- F4: a zip member's own comment -------------------------------------------------------


def _member_comment_zip(comment: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        info = zipfile.ZipInfo("paper.txt")
        info.comment = comment
        archive.writestr(info, b"clean words")
    return out.getvalue()


def test_f4_zip_member_comment_meets_every_rule() -> None:
    assert _scan(_member_comment_zip(HOME.encode()), {}) is True
    assert _scan(_member_comment_zip(b"deploy/.env"), {}) is True
    assert _scan(_member_comment_zip(b"changes.environmental"), {}) is False  # control


# --- F5: a UTF-16 word holding .env is no env file, in either byte order -----------------


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("word", [".envx", "changes.envy", "changes.environmental", "x.env9"])
@pytest.mark.parametrize("after", [b"", b"A", b"\x00", b"\xd8"], ids=["end", "letter", "nul", "surrogate"])
def test_f5_a_utf16_word_holding_env_passes(word: str, encoding: str, after: bytes) -> None:
    assert _scan(word.encode(encoding) + after, {}) is False


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("name", [".env", "prod.env", '".env"', ".env.local"])
def test_f5_the_utf16_env_file_name_is_still_refused(name: str, encoding: str) -> None:
    assert _scan(("see " + name + " now").encode(encoding), {}) is True
