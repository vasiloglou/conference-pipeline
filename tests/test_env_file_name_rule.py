"""One ".env" file-name rule for every scan and every file type (PR 49, after pr-49-r6).

".env" was defined five ways (a substring for text, narrower rules for PDF text and raw PDF
bytes, a separator rule for binary bytes) and the hub and release check skipped .env
findings in PDFs and zips; each seam leaked in review. Now one rule decides: ".env" is the
file name unless a letter or digit continues it. These cases put the same names in every
carrier and require the same verdict from each, and from both publication gates.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tests.test_release_cli import SLUG, _no_secrets, cli, w0, w4  # noqa: F401
from tests.test_release_reader_families import _ascii_pdf, _pdf_parts

FILE_NAMES = [".env", "prod.env", '".env"', "(.env)", ".env.local", "srv/app/.env", "see .env, then"]
WORDS = ["changes.environmental", "the .envelope", "plain public text"]


def _zip(name: str, data: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, data)
    return out.getvalue()


def _pdf_string(text: str) -> bytes:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode()


CARRIERS = {
    "text": lambda t: f"intro {t} outro\n".encode(),
    "utf16_text": lambda t: f"intro {t} outro\n".encode("utf-16"),
    "pdf_page": lambda t: _ascii_pdf(f"intro {t} outro"),
    "pdf_compressed_page": lambda t: _pdf_parts(b"BT /F1 10 Tf 10 100 Td (" + _pdf_string(t) + b") Tj ET"),
    "pdf_comment": lambda t: _ascii_pdf("hello") + f"% {t}\n".encode(),
    "pdf_xmp": lambda t: _pdf_parts(b"BT /F1 10 Tf 10 100 Td (hello) Tj ET", xmp=f"<p>{t}</p>".encode()),
    "zip_member": lambda t: _zip("notes.txt", f"intro {t} outro\n".encode()),
    "zip_of_pdf": lambda t: _zip("paper.pdf", _ascii_pdf(f"intro {t} outro")),
}


def _flagged(data: bytes) -> bool:
    from conference_pipeline._security.private_markers import builtin_rules, has_publishing_marker_bytes

    return has_publishing_marker_bytes(data, rules=builtin_rules())


@pytest.mark.parametrize("carrier", sorted(CARRIERS))
@pytest.mark.parametrize("name", FILE_NAMES)
def test_an_env_file_name_is_refused_in_every_carrier(carrier: str, name: str) -> None:
    assert _flagged(CARRIERS[carrier](name)) is True


@pytest.mark.parametrize("carrier", sorted(CARRIERS))
@pytest.mark.parametrize("word", WORDS)
def test_a_word_holding_env_passes_in_every_carrier(carrier: str, word: str) -> None:
    assert _flagged(CARRIERS[carrier](word)) is False


def test_text_rules_agree_with_the_byte_rule() -> None:
    from conference_pipeline._security.private_markers import has_publishing_marker, publishing_literal_rules

    env_rule = next(r for r in publishing_literal_rules() if r.id == ".env")
    for name in FILE_NAMES:
        assert has_publishing_marker(name, rules=()) and env_rule.pattern.search(name), name
    for word in WORDS:
        assert not has_publishing_marker(word, rules=()) and not env_rule.pattern.search(word), word


def test_opaque_bytes_need_a_path_separator() -> None:
    # Four bytes collide by chance in compressed or binary data: there, a path names the file.
    assert _flagged(b"\x00\xff/srv/app/.env\x81") is True
    assert _flagged(b"\x00\xff.env\x81") is False


@pytest.mark.parametrize("suffix,make", [("txt", CARRIERS["text"]), ("pdf", CARRIERS["pdf_page"]), ("pdf", CARRIERS["pdf_comment"])])
@pytest.mark.parametrize("text,refused", [('".env"', True), ("prod.env", True), ("changes.environmental", False)])
def test_hub_and_release_check_give_one_verdict(tmp_path: Path, monkeypatch, w4, suffix, make, text, refused) -> None:
    from conference_pipeline.publishing.release_hub import _marker_findings
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path / "hub")
    (hub / f"paper.{suffix}").write_bytes(make(text))
    assert bool([f for f in _marker_findings(hub, {}) if f"paper.{suffix}" in f]) is refused

    (w4.bundle / "decks" / "01_sample" / f"deck.{suffix}").write_bytes(make(text))
    cli().release_check(SLUG, env={}, secret_scanner=_no_secrets, **w4.kw())
    assert any("private_marker" in line and f"deck.{suffix}" in line for line in w4.out) is refused, w4.text()
