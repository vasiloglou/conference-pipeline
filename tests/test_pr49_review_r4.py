"""v6.0 Phase 7.1, PR 49 review pr-49-r4: one regression test per finding (F1-F2)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tests.test_release_cli import SLUG, _no_secrets, cli, w0, w4  # noqa: F401
from tests.test_release_reader_families import _ascii_pdf, _pdf_parts

MARKER = "codename-zephyr-fixture"
ENV = {"CONFERENCE_PIPELINE_PRIVATE_MARKERS": MARKER}
HOME = "/" + "home/private-review-user/internal-note"


def _compressed_binary_pdf() -> bytes:
    # A readable page, and a Flate-compressed binary stream that text extraction never parses.
    data = _pdf_parts(b"BT /F1 10 Tf 10 100 Td (hello) Tj ET", xmp=b"\x00\xff" + MARKER.encode() + b"\x81")
    assert MARKER.encode() not in data
    return data


def _zipped(name: str, data: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, data)
    return out.getvalue()


def _commented_pdf() -> bytes:
    # A clean ASCII PDF with a PDF comment appended after %%EOF: offsets stay valid.
    return _ascii_pdf("Adaptive pipelines") + f"% {HOME}\n".encode()


# --- F1: extra markers in a decoded binary PDF stream -----------------------------------


@pytest.mark.parametrize("wrap", [False, True])
def test_f1_extra_marker_in_a_compressed_binary_pdf_stream(wrap: bool) -> None:
    from conference_pipeline._security.private_markers import extra_rules_from_env, has_publishing_marker_bytes

    data = _compressed_binary_pdf()
    if wrap:
        data = _zipped("paper.pdf", data)
    assert has_publishing_marker_bytes(data, rules=extra_rules_from_env(ENV)) is True
    # Control: the same PDF without the configured marker is clean.
    assert has_publishing_marker_bytes(data, rules=extra_rules_from_env({})) is False


def test_f1_hub_refuses_the_compressed_binary_pdf(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing.release_hub import _marker_findings
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "paper.pdf").write_bytes(_compressed_binary_pdf())
    assert [f for f in _marker_findings(hub, ENV) if "paper.pdf" in f]


def test_f1_release_check_refuses_the_compressed_binary_pdf(w4) -> None:
    (w4.bundle / "decks" / "01_sample" / "deck.pdf").write_bytes(_compressed_binary_pdf())
    cli().release_check(SLUG, env=ENV, secret_scanner=_no_secrets, **w4.kw())
    assert any("private_marker" in line and "deck.pdf" in line for line in w4.out), w4.text()


# --- F2: a private path in a PDF comment is still a marker ------------------------------


def test_f2_hub_flags_a_home_path_in_a_pdf_comment(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.publishing.release_hub import _marker_findings
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    (hub / "paper.pdf").write_bytes(_commented_pdf())
    assert [f for f in _marker_findings(hub, {}) if "paper.pdf" in f]
    # Control: the same PDF without the comment, and one naming changes.environmental, pass.
    (hub / "paper.pdf").write_bytes(_ascii_pdf("Adaptive pipelines and changes.environmental changes"))
    assert not [f for f in _marker_findings(hub, {}) if "paper.pdf" in f]


def test_f2_release_check_flags_a_home_path_in_a_pdf_comment(w4) -> None:
    (w4.bundle / "decks" / "01_sample" / "deck.pdf").write_bytes(_commented_pdf())
    cli().release_check(SLUG, env={}, secret_scanner=_no_secrets, **w4.kw())
    assert any("private_marker" in line and "deck.pdf" in line for line in w4.out), w4.text()
